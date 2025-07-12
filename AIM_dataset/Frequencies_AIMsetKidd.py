
import json
#import io
import os
import glob
import numpy as np
import pandas as pd
from cyvcf2 import VCF
import math


def vcf_to_df(vcf_file):
    vcf = VCF(vcf_file)
    samples = vcf.samples
    records = []

    for variant in vcf:
        record = [str(variant.POS)]
        record.extend(f"{variant.genotypes[i][0]}|{variant.genotypes[i][1]}" if variant.genotypes[i][2] == True else -1 for i in range(len(samples)))
        records.append(record)

    df = pd.DataFrame(records, columns=['marker'] + samples)
    df = df.set_index('marker').astype('category')
    #print(df)
    return df


def load_data(data_path_vcf, data_path_y) -> [pd.DataFrame, pd.Series]:
    """Read all vcf files and create a Dataframe in the right format for TabPFN-
    Classification. And create a series with the matching y-values (populations)."""

    #files = glob.glob(os.path.join(data_path_vcf, "*.vcf"))
    data = vcf_to_df(data_path_vcf)
    data = data.T

    # read in the y-values
    populations = pd.read_csv(data_path_y, sep="\t", header=None)
    populations = populations.set_index(populations.columns[0])
    y = populations.loc[data.index] # only evaluate the populations occurring in the files
    y = y[1]

    return data, y


def compute_allelefrequencies(X: pd.DataFrame, y: pd.Series, populations: list[str]) -> dict[str, dict[str, float]]:
    """Compute the allelefrequencies for each population from a given Dataframe with DNA data"""

    frequencies = {}
    for pop in populations:
        marker_frequencies = {}
        indices = y[y == pop].index
        X_pop = X.loc[indices]
        # Schneller: Zähle '1' in allen Genotypen direkt vektorisert für alle Marker
        allele_counts = X_pop.apply(lambda col: col.str.count('1'), axis=0).sum(axis=0)
        # Frequenz berechnen: (Summe der '1'-Allele) / (Anzahl Individuen * 2)
        marker_frequencies = (allele_counts / (len(indices) * 2)).to_dict()
        #print(marker_frequencies)

        frequencies[pop] = marker_frequencies

    return frequencies

if __name__ == "__main__":
    data_path_vcf = "1000G_AIMsetKidd.vcf"  # vcf file with DNA data
    data_path_populations = "1000G_SampleListWithLocations.txt"  # csv file with y-values (populations) for all three data sets

    data, y = load_data(data_path_vcf, data_path_populations)

    populations = y.unique().tolist()
    #n = len(populations)
    print("Computation of frequencies...")
    frequencies = compute_allelefrequencies(data, y, populations)
    freq_df = pd.DataFrame(frequencies)
    freq_df.to_csv("Frequencies_1000G_AIMsetKidd.csv")
