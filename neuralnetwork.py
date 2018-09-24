import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def cross_validation(layers,activations):
    cv_score = pd.DataFrame(index=layers, columns=activations)
    score = pd.DataFrame(index=layers, columns=activations)
    computation_time = pd.DataFrame(index=layers, columns=activations)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

    best_accuracy = 0
    best_l = 0
    best_a = ''

    for l in layers:
        for a in activations:
            start=time.time()
            clf = MLPClassifier(random_state=rs, hidden_layer_sizes=l, activation=a)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            end=time.time()
            computation_time.at[l,a] = end - start
            score.at[l, a] = accuracy_score(y_test, y_predict)
            if best_accuracy<score.at[l, a]:
                best_accuracy = score.at[l, a]
                best_l=l
                best_a=a
            cv_score.at[l, a]=np.mean(cross_val_score(clf, X, y, cv=3, scoring='accuracy'))
    draw_cross_validation(layers, activations, cv_score, score, 'Neural Network')
    draw_computing_time(layers, activations, computation_time, 'Neural Network')

    return best_l,best_a

def draw_cross_validation(layers, activations, cv_score, score, model):
    for a in activations:
        color = np.random.rand(3, )
        plt.plot(layers, cv_score[a], color=color, label="min_samples_split="+str(a)+' (Cross Validation)')
        plt.plot(layers, score[a], '--', color=color, label="min_samples_split=" + str(a))
    plt.legend()
    plt.xlabel('Hidden Layers for '+model)
    plt.ylabel('Accuracy')
    plt.show()

def draw_computing_time(layers, activations, computation_time,model):
    for a in activations:
        plt.plot(layers, computation_time[a], label="Computing Time for activations="+str(a))
    plt.legend()
    plt.xlabel('Hidden Layers for '+model)
    plt.ylabel('Computing Time')
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
    plt.legend(loc="best")
    return plt

if __name__ == "__main__":

    data_selection = 'CreditCard'
    #data_selection = 'SeasonsStats'

    if data_selection == 'CreditCard':
        dataset = pd.read_csv("CreditCard.csv")
        X = dataset.drop('default_payment', axis=1)
        y = dataset['default_payment']

    if data_selection == 'SeasonsStats':
        dataset = pd.read_csv("SeasonsStats.csv")
        X = dataset.drop('Pos', axis=1)
        y = dataset['Pos']

    rs = 1

    #classifiers = [DecisionTreeClassifier(random_state=rs),MLPClassifier(random_state=rs),KNeighborsClassifier(),SVC(kernel="linear", C=0.025, random_state=rs),AdaBoostClassifier(random_state=rs)]

    best_params = cross_validation(layers=range(10,201,50), activations=['identity', 'logistic', 'tanh', 'relu'])

    clf = MLPClassifier(random_state=rs, hidden_layer_sizes=best_params[0], activation=best_params[1])
    title = "Learning Curves (Neural Network) with hidden_layer_sizes = "+str(best_params[0])+" activation = "+str(best_params[1])
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(clf, title, X, y, ylim=(0.1, 1.1), cv=cv, n_jobs=4)
    plt.show()