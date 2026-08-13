import csv
import gzip
import subprocess


# ============================================================
# File names
# ============================================================

vcf_file = "EUR.MAF01.biallelicSNP.allchr.vcf.gz"
panel_file = "integrated_call_samples_v3.20130502.ALL.panel"
output_file = "EUR.MAF01.biallelicSNP.allchr.dosage.csv.gz"


# ============================================================
# Read sample IDs from the VCF file
# ============================================================

result = subprocess.run(
    ["bcftools", "query", "-l", vcf_file],
    capture_output=True,
    text=True,
    check=True,
)

samples = result.stdout.strip().splitlines()

print(f"Found {len(samples)} individuals in the VCF file.")


# ============================================================
# Read population information from the 1000 Genomes panel file
# ============================================================

population = {}

with open(panel_file, "r") as f:
    for line in f:
        fields = line.strip().split()

        if len(fields) >= 3:
            sample_id = fields[0]
            pop = fields[1]
            super_pop = fields[2]

            # Keep only European populations
            if super_pop == "EUR":
                population[sample_id] = pop


# ============================================================
# Check that population information exists for every individual
# ============================================================

missing_population = [
    sample
    for sample in samples
    if sample not in population
]

if missing_population:
    raise ValueError(
        f"Population information is missing for "
        f"{len(missing_population)} samples: "
        f"{missing_population[:10]}"
    )


# ============================================================
# Convert VCF genotypes to dosage values
#
# 0 = homozygous reference
# 1 = heterozygous
# 2 = homozygous alternative
# NA = missing genotype
# ============================================================

def genotype_to_dosage(gt):
    if gt in ("0|0", "0/0"):
        return 0

    if gt in ("0|1", "1|0", "0/1", "1/0"):
        return 1

    if gt in ("1|1", "1/1"):
        return 2

    return "NA"


# ============================================================
# Start bcftools query
#
# For every SNP, extract:
# - chromosome
# - position
# - genotype of every individual
#
# The output is streamed instead of being loaded completely
# into memory.
# ============================================================

process = subprocess.Popen(
    [
        "bcftools",
        "query",
        "-f",
        r"%CHROM\t%POS[\t%GT]\n",
        vcf_file,
    ],
    stdout=subprocess.PIPE,
    text=True,
)


# ============================================================
# Write the compressed CSV file
# ============================================================

with gzip.open(output_file, "wt", newline="") as f:
    writer = csv.writer(f)

    # First row: individual identifiers
    writer.writerow(
        ["Position"] + samples
    )

    # Second row: population of each individual
    writer.writerow(
        ["Population"] + [population[s] for s in samples]
    )

    n_snps = 0

    # Process one SNP at a time
    for line in process.stdout:
        fields = line.rstrip("\n").split("\t")

        chromosome = fields[0]
        position = fields[1]
        genotypes = fields[2:]

        # Use chromosome:position as a unique SNP identifier
        snp_position = f"{chromosome}:{position}"

        # Convert all genotypes to 0, 1, 2, or NA
        dosages = [
            genotype_to_dosage(gt)
            for gt in genotypes
        ]

        # Write one SNP per row
        writer.writerow(
            [snp_position] + dosages
        )

        n_snps += 1

        # Print progress every 100,000 SNPs
        if n_snps % 100000 == 0:
            print(f"{n_snps:,} SNPs written")


# ============================================================
# Check whether bcftools finished successfully
# ============================================================

return_code = process.wait()

if return_code != 0:
    raise RuntimeError(
        f"bcftools query failed with exit code {return_code}"
    )


# ============================================================
# Final summary
# ============================================================

print(f"Finished: {n_snps:,} SNPs")
print(f"Output file: {output_file}")
