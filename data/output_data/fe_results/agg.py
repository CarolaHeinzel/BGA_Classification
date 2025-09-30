import pandas as pd
import glob
import os

# Path to your folder
folder_path = "./"  # Change this to your folder path

# Pattern for matching files
file_pattern = os.path.join(folder_path, "ag_feature_importance_*.csv")

# Collect all matching files
files = glob.glob(file_pattern)

# Initialize an empty DataFrame
agg_df = pd.DataFrame()

# Process each file
for file in files:
    df = pd.read_csv(file, index_col=0)  # Index is the feature ID
    # If agg_df is empty, initialize with this file
    if agg_df.empty:
        agg_df = df[['importance']].copy()
        agg_df.rename(columns={'importance': f"importance_{os.path.basename(file)}"}, inplace=True)
    else:
        agg_df = agg_df.join(df[['importance']].rename(columns={'importance': f"importance_{os.path.basename(file)}"}), how='outer')

# Replace NaN with 0 for aggregation
agg_df = agg_df.fillna(0)

# Compute aggregated importance (sum across all files)
agg_df['importance_sum'] = agg_df.sum(axis=1)

# Or compute mean importance
agg_df['importance_mean'] = agg_df.mean(axis=1)

# Sort by sum (or mean) in descending order
agg_df = agg_df.sort_values('importance_sum', ascending=False)

# Save aggregated results
output_path = os.path.join(folder_path, "aggregated_feature_importance.csv")
agg_df.to_csv(output_path)

print(f"Aggregated feature importance saved to {output_path}")