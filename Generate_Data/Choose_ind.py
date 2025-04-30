import pandas as pd
# Determines the names of the individuals that we are interested in
# The 1000G.txt file is from Peter Pfaffelhuber's GitHub page

path  = "1000G.txt"
df = pd.read_csv(path, sep="\t", header=None)

df_filtered = df[df[1].isin(['YRI', 'ESN','MSL', 'LWK', 'GWD', 'ASW', 'ACB'])]

# This is the list that we use as input file for the .sh script
df_filtered[0].to_csv("AFR.txt", sep="\t", index=False, header=False)
#%%
unique_values = df.loc[df[2] == "AMR", 1].unique()
print(unique_values)
