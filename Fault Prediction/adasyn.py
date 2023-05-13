import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.impute import SimpleImputer



# Load the concatenated dataset with PCA
df = pd.read_csv('concatenated_dataset_pca.csv')

# Drop rows containing NaN or infinity values
df = df.dropna()

# Separate features and target variable
X = df.drop('bugs', axis=1)
y = df['bugs']

# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
#
# # Drop rows containing '-'
# X_imputed = X_imputed[~X_imputed.isin(['nan']).any(axis=1)]

# Apply ADASYN method to handle class imbalance
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Save the resampled dataset
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
df_resampled.to_csv('concatenated_dataset_pca_adasyn.csv', index=False)
