# Data preprocessing
# Importing the libraries

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding the Independent variable column named Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encoding the Independent variable column named Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN
# Initializing the ANN

ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding a second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN
# Compiling the ANN

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the Ann on the training set

ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Making single prediction

print (ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 5000]])) > 0.5)

# Predicting the test result

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


