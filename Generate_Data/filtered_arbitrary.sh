for chr in {1..22}; do
  file="EUR_only_ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.recode.vcf"
  vcftools --vcf ${file} --thin 2000 --min-alleles 2 --non-ref-ac 2 --recode --out filtered_${file}
  vcftools --vcf filtered_${file}.recode.vcf --maf 0.05 --recode --out maf_filtered_chr${chr}
done
bcftools concat -O z -o combined_filtered.vcf.gz maf_filtered_chr{1..22}.recode.vcf
