"""Code for plotting the results from our experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_metric(df, metric):
    """Plots a bar chart comparing different results from TabPFN
    classification based on the specified metric."""
    # Ensure the 'plots' directory exists
    Path("plots").mkdir(parents=True, exist_ok=True)

    # Create the barplot
    plt.figure(figsize=(6, 6))
    #sns.barplot(data=df, x="Model", y=metric, hue="Model")
    sns.barplot(data=df, x="Data", y=metric, hue="Data")
    #sns.barplot(data=df_2, x="Data", y=metric, hue="Model")
    plt.xlabel("Data", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()

    #########################################
    # mean values of the metrics
    mean_values = df.groupby("Data")[metric].mean().reset_index()
    print(mean_values)
    #########################################

    # Save the plot to file
    plot_path = Path("plots") / f"{metric.replace(' ', '_')}_comparison.pdf"
    plt.savefig(plot_path)
    plt.show()



if __name__ == "__main__":

    # data to compare
    df_allele = pd.read_csv(r"results/results_TabPFN_vcf_allele.csv")
    df_random = pd.read_csv(r"results/results_TabPFN_vcf_random.csv")
    df_filtered_population_eur = pd.read_csv("results/results_TabPFN_filtered_population_eur.csv")

    df_allele["Data"] = ["allele" for _ in range(50)]
    df_random["Data"] = ["random" for _ in range(50)]
    df_filtered_population_eur["Data"] = ["filtered_population_eur" for _ in range(50)]

    df = pd.concat([df_allele, df_random, df_filtered_population_eur])
    #print(df)

    metrics = ["Accuracy", "Balanced Accuracy", "ROC AUC", "Log Loss"]
    for metric in metrics:
        plot_metric(df, metric)
