import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def cross_validation(neighbors,ps):
    cv_score = pd.DataFrame(index=neighbors, columns=ps)
    score = pd.DataFrame(index=neighbors, columns=ps)
    computation_time = pd.DataFrame(index=neighbors, columns=ps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

    best_accuracy = 0
    best_n = 0
    best_p = 0

    for n in neighbors:
        for p in ps:
            start = time.time()
            clf = KNeighborsClassifier(n_neighbors=n, p=p)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            end = time.time()
            computation_time.at[n, p] = end - start
            score.at[n, p] = accuracy_score(y_test, y_predict)
            if best_accuracy<score.at[n, p]:
                best_accuracy = score.at[n, p]
                best_n=n
                best_p=p
            cv_score.at[n, p]=np.mean(cross_val_score(clf, X, y, cv=10, scoring='accuracy'))
    draw_cross_validation(neighbors, ps, cv_score, score, 'K Nearest Neighbors')
    draw_computing_time(neighbors, ps, computation_time, 'K Nearest Neighbors')

    return best_n, best_p

def draw_cross_validation(neighbors, ps, cv_score, score, model):
    for p in ps:
        color = np.random.rand(3, )
        plt.plot(neighbors, cv_score[p], color=color, label="P="+str(p)+' (Cross Validation)')
        plt.plot(neighbors, score[p], '--', color=color, label="P=" + str(p))
        plt.legend()
    plt.xlabel('Nearest Neighbors for '+model)
    plt.ylabel('Accuracy')
    plt.show()

def draw_computing_time(neighbors, ps, computation_time, model):
    for p in ps:
        plt.plot(neighbors, computation_time[p], label="Computing Time for p="+str(p))
    plt.legend()
    plt.xlabel('Nearest Neighbors for '+model)
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

    best_params = cross_validation(neighbors=list(range(2, 10, 3))+list(range(20,51,15)),ps=range(1,6,1))

    clf = KNeighborsClassifier(n_neighbors=best_params[0],p=best_params[1])
    title = "Learning Curves (K Nearest Neighbour) with n_neighbors = "+str(best_params[0])+" p = "+str(best_params[1])
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(clf, title, X, y, ylim=(0.1, 1.1), cv=cv, n_jobs=4)
    plt.show()