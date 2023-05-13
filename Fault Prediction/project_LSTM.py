import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout, Flatten, concatenate, Permute, Multiply, Lambda
from keras.callbacks import EarlyStopping
from keras.layers import Activation, RepeatVector, TimeDistributed, Reshape
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# set the path to the folder containing the CSV files
folder_path = 'promise_data'

# create an empty list to store dataframes
dfs = []

# loop through each folder in the directory
for folder in os.listdir(folder_path):
    # check if it's a directory
    if os.path.isdir(os.path.join(folder_path, folder)):
        # loop through each CSV file in the folder
        for filename in os.listdir(os.path.join(folder_path, folder)):
            if filename.endswith('.csv'):
                # read the CSV file as a dataframe
                df = pd.read_csv(os.path.join(folder_path, folder, filename))
                # append the dataframe to the list
                dfs.append(df)

# concatenate all dataframes in the list into a single dataframe
concatenated_df = pd.concat(dfs)

# save the concatenated dataframe to a CSV file
concatenated_df.to_csv('concatenated_dataset.csv', index=False)

# -----------------------------------------------------------------------------------------------------
#                                               PCA

# Load concatenated dataset
df = pd.read_csv("concatenated_dataset.csv")

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


# -----------------------------------------------------------------------------------------------------
#                                               adasyn

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


# -----------------------------------------------------------------------------------------------------
#                                               LSTM Model

# Load preprocessed dataset
data = pd.read_csv("concatenated_dataset_pca_adasyn.csv")

# Remove NaN values
data = data.dropna()

# Split dataset into input and output features
X = data.drop(columns=['bugs'])
y = data['bugs']

# Standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input features for LSTM layer
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

timesteps = 1
features = 24
nodes = 150
epochs = 1000

# Input layer
inputs = Input(shape=(timesteps, features))

# Bidirectional LSTM layer
lstm_out1 = Bidirectional(LSTM(nodes, return_sequences=True))(inputs)

# Bidirectional GRU layer
gru_out = Bidirectional(GRU(nodes, return_sequences=True))(lstm_out1)

# LSTM layer
lstm_out2 = LSTM(nodes, return_sequences=True)(gru_out)

# Attention mechanism
attention = Dense(1, activation='tanh')(lstm_out2)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(nodes)(attention)
attention = Permute([2, 1])(attention)

# Apply attention weights
merged = Multiply()([lstm_out2, attention])
merged = Lambda(lambda xin: K.sum(xin, axis=1))(merged)

# Dropout layer
merged = Dropout(0.5)(merged)

# Output layer
output_layer = Dense(1, activation='sigmoid')(merged)

# Define model
model = Model(inputs=inputs, outputs=output_layer)

# Compile model
# optimizer = Adam(learning_rate=0.01)
# Convert the optimizer to its string identifier
# optimizer_name = optimizer.__class__.__name__
# model.compile(loss='binary_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Fit model to training data
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])

# Evaluate model on testing data
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
print("-----------------------------")
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Obtain the train accuracy from history
train_accuracy = history.history['accuracy'][-1]

# Print the results
print('Train accuracy:', train_accuracy*100,"%")
print('Test accuracy:', test_accuracy*100,"%")

# make predictions on test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)



