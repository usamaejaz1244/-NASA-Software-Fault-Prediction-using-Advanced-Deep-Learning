import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Load concatenated dataset
df = pd.read_csv("concatenated_dataset.csv")

# # Replace the missing values or special characters with NaN
# df = df.replace('-', np.nan)
#
# # Fill the missing values with the mean value of the column
# df = df.fillna(df.mean())

# Drop rows containing '-'
df = df[~df.isin(['-']).any(axis=1)]

# Drop non-numeric columns
df = df.drop(['name'], axis=1)

# Split data and target values
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Create PCA object with desired number of components
pca = PCA(n_components=24)

# Fit and transform data
X_pca = pca.fit_transform(X)

# Save reduced dataset
df_pca = pd.DataFrame(X_pca, columns=["PC"+str(i) for i in range(1,25)])
df_pca["bugs"] = y
df_pca.to_csv("concatenated_dataset_pca.csv", index=False)
