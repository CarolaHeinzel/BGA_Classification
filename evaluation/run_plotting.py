"""Code for plotting the results from our experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_metric(df, metric, model):
    """Plots a bar chart comparing different results from TabPFN
    classification based on the specified metric."""

    # Ensure the 'plots' directory exists
    path_for_fig = Path(__file__).parent / "data" / "output_data" / "plots"
    path_for_fig.mkdir(parents=True, exist_ok=True)

    # Create the barplot
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x="Name", y=metric, hue="Name")
    #plt.xlabel("Feature Selection Method", fontsize=12)
    plt.xlabel("Feature Selection Method", fontsize=20)
    #plt.xticks(rotation=45, fontsize=10)
    plt.xticks(rotation=45, fontsize=17)
    #plt.ylabel(metric, fontsize=12)
    plt.ylabel(metric, fontsize=20)

    if metric != "": #"Log Loss":
        y_min = df[metric].min()
        margin = 0.05 * (df[metric].max() - y_min)
        ax.set_ylim(y_min - margin, None)
    plt.tight_layout()

    # Save the plot to file
    plot_path = path_for_fig / f"{model}_{metric.replace(' ', '_')}.pdf"
    plt.savefig(plot_path)
    plt.show()


def compute_metric(df, metric):
    # mean values of the metrics
    mean_values = df.groupby("Name")[metric].mean().reset_index()
    print(mean_values)


if __name__ == "__main__":

    data_path = Path(__file__).parent / "data" / "output_data"

    # data to compare
    df_allele = pd.read_csv(data_path / "results_allele.csv") # allele
    df_allele["Name"] = ["Allele Freq. Method,\n100 Features" for _ in range(120)]

    df_baseline = pd.read_csv(data_path / "baseline_results_neu.csv") # AGFeatures, StaticExpert
    df_baseline["Name"] = ["New Method,\n 100 Features" if df_baseline["Model"][i].split("_")[0]=="AGFeaturesC"
                           else "Expert knowledge,\n104 Features" for i in range(240)]

    df_baseline_alt = pd.read_csv(data_path / "baseline_results_wo_tabm.csv") # random, all
    df_baseline_alt = df_baseline_alt[df_baseline_alt["Model"].isin(["TabPFN", "NaiveBayes"])].reset_index() #"RandomFeatures_TabPFN", "RandomFeatures_NaiveBayes",
    df_baseline_alt = df_baseline_alt.drop(columns="index")
    df_baseline_alt["Name"] = ["All Features" for _ in range(120)]
    #df_baseline_alt["Name"] = [df_baseline_alt["Model"][i].split("_")[0] if len(df_baseline_alt["Model"][i].split("_")[0]) > 12 else "All features" for i in range(240)]

    df_200 = pd.read_csv(data_path / "baseline_results_200features.csv") # AGFeatures 200
    df_200 = df_200[df_200["Model"].isin(["AGFeaturesC_TabPFN", "AGFeaturesC_NaiveBayes"])]
    df_200["Name"] = ["New Method,\n200 Features" for _ in range(120)]

    df_50 = pd.read_csv(data_path / "baseline_results_50features.csv") # AGFeatures 50
    df_50 = df_50[df_50["Model"].isin(["AGFeaturesC_TabPFN", "AGFeaturesC_NaiveBayes"])]
    df_50["Name"] = ["New Method,\n50 Features" for _ in range(120)]

    df = pd.concat([df_baseline, df_allele, df_baseline_alt, df_200, df_50]).reset_index()
    df = df.drop(columns="index")

    metrics = ["Accuracy", "Balanced Accuracy", "ROC AUC", "Log Loss"]
    for model in ["TabPFN", "NaiveBayes"]:
        df_filtered = df[df["Model"].str.contains(model, na=False)]
        print(df_filtered)
        for metric in metrics:
            compute_metric(df_filtered, metric)
            plot_metric(df_filtered, metric, model)
