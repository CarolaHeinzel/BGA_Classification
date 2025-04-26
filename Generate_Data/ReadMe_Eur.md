### Download data
./download.sh

### Download sampling location list from
### European samples are from CEU, IBS, GBR, FIN, TSI

https://www.internationalgenome.org/data-portal/sample
call it igsr_samples.tsv

### Extract European samples
./extract_eur.sh

Now we follow
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/admixture_files/README.admixture_20141217


### Filter bi-allelic SNPs which are 2k apart
./filtered.sh

