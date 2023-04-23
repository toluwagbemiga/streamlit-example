from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('diabetes_pre_type2.csv')
data2 = pd.read_csv('diabetes_pre_type2.csv')
data
data.drop(data[data.Class_012 == 2].index, inplace=True)
data['Class_012'].value_counts()
data2.drop(data2[data2.Class_012 == 1].index, inplace=True)
data2['Class_012'].value_counts()
count_classes = pd.value_counts(data['Class_012'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Prediabetes histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
count_classes = pd.value_counts(data2['Class_012'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Diabetes type 2 histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
X = data.iloc[:, :-1].values
Y = data2.iloc[:, :-1].values
data_class_outcomes = data['Class_012']
data.drop(['Class_012'], axis = 1, inplace = True)
data_class_outcomes2 = data2['Class_012']
data2.drop(['Class_012'], axis = 1, inplace = True)
#Splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, data_class_outcomes, test_size=0.3, random_state=42)
# Instantiate a Random Forest Classifier object
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
# Train the model on the training set
classifier.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Draw confusion matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='', cmap='Blues', annot_kws={'size': 16, 'ha': 'center', 'va': 'center'}, cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.title('Confusion matrix', fontsize=16)
plt.show()

# Plot feature importance

importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
names = [data.columns[i] for i in indices]
plt.figure()
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()

tn, fp, fn, tp = cm.ravel()
print('True positives:', tp)
print('True negatives:', tn)
print('False positives:', fp)
print('False negatives:', fn)

#Splitting the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data2, data_class_outcomes2, test_size=0.3, random_state=42)
# Instantiate a Random Forest Classifier object
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
# Train the model on the training set
classifier.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='', cmap='Blues', annot_kws={'size': 16, 'ha': 'center', 'va': 'center'}, cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.title('Confusion matrix', fontsize=16)
plt.show()

# Plot feature importance

importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
names = [data2.columns[i] for i in indices]
plt.figure()
plt.title('Feature Importance')
plt.bar(range(Y.shape[1]), importances[indices])
plt.xticks(range(Y.shape[1]), names, rotation=90)
plt.show()

tn, fp, fn, tp = cm.ravel()
print('True positives:', tp)
print('True negatives:', tn)
print('False positives:', fp)
print('False negatives:', fn)

