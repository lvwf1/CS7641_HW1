import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score

h = .02  # step size in the mesh

names = ["Decision Tree", "Neural Net", "Nearest Neighbors", "Linear SVM", "GradientBoosting"]

data_selection = 'CreditCard'
# data_selection = 'SeasonsStats'

if data_selection == 'CreditCard':
    dataset = pd.read_csv("CreditCard.csv", skiprows=random.sample(range(1, 30000), 25000))
    X = dataset.drop('default_payment', axis=1)
    y = dataset['default_payment']

if data_selection == 'SeasonsStats':
    dataset = pd.read_csv("SeasonsStats.csv", skiprows=random.sample(range(1, 15095), 10000))
    X = dataset.drop('Pos', axis=1)
    y = dataset['Pos']

rs = 1
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=rs)

classifiers = [
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(hidden_layer_sizes=100),
    KNeighborsClassifier(n_neighbors=35),
    SVC(C=1.0, kernel='rbf'),
    GradientBoostingClassifier(n_estimators=10)]

performance_matrix=[]
runtime_matrix=[]
for clf in classifiers:
    start=time.time()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    end=time.time()
    print('Accuracy Score：', accuracy_score(y_test, y_predict))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_predict, y_test))
    print('Mean Squared Error:', metrics.mean_squared_error(y_predict, y_test))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_predict, y_test)))
    performance_matrix.append([accuracy_score(y_test, y_predict),metrics.mean_absolute_error(y_predict, y_test),metrics.mean_squared_error(y_predict, y_test),np.sqrt(metrics.mean_squared_error(y_predict, y_test))])
    runtime_matrix.append(end-start)

# set width of bar
barWidth = 0.1

# set height of bar
bars1 = performance_matrix[0]
bars2 = performance_matrix[1]
bars3 = performance_matrix[2]
bars4 = performance_matrix[3]
bars5 = performance_matrix[4]

# Set position of bar on X axis
r2 = np.arange(len(bars2))
r1 = [x - barWidth for x in r2]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Make the plot
plt.bar(r1, bars1, color='#da45e7', width=barWidth, edgecolor='white', label='Decision Tree')
plt.bar(r2, bars2, color='#658f0a', width=barWidth, edgecolor='white', label='Neural Net')
plt.bar(r3, bars3, color='#9c3ef0', width=barWidth, edgecolor='white', label='Nearest Neighbors')
plt.bar(r4, bars4, color='#470cf1', width=barWidth, edgecolor='white', label='SVM')
plt.bar(r5, bars5, color='#1fb963', width=barWidth, edgecolor='white', label='Gradient Boosting')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ["Accuracy Score", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error"])

# Create legend & Show graphic
plt.legend()
plt.show()

plt.bar(np.arange(5), runtime_matrix)
plt.xticks(np.arange(5), ("Decision Tree", "Neural Net", "Nearest Neighbors", "Linear SVM", "GradientBoosting"))
plt.show()