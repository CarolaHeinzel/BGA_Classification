wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502_phase3_samples.tsv
awk '$6 == "EUR" {print $1}' igsr_samples.tsv > european_samples.txt
for chr in {1..22}; do
  file="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
  vcftools --gzvcf ${file} --keep european_samples.txt --recode --recode-INFO-all --out EUR_only_${file}
done
