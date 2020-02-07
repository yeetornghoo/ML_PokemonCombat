import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def run(data_model):

    clf_dict = {
        'random_forest': RandomForestClassifier(n_estimators=100),
        'OneVsOneClassifier': OneVsOneClassifier(MultinomialNB()),
        'svn': svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                       tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                       decision_function_shape='ovr', break_ties=False, random_state=None),
        'LinearSVC': LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                               fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                               max_iter=100000),
        'Logistic': LogisticRegression(random_state=42, max_iter=10000, multi_class='auto', solver='saga'),
        'MultinomialNB': MultinomialNB(),
        'GaussianNB': GaussianNB(),
        'CART': DecisionTreeClassifier(),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100),
        'ANN': MLPClassifier(solver='adam', hidden_layer_sizes=(70, 30), random_state=42, activation='relu',
                             max_iter=200000, learning_rate='invscaling'),
        'ada_boost': AdaBoostClassifier(n_estimators=100)
    }

    for name, clf in clf_dict.items():
        model = clf.fit(data_model.X_train, data_model.y_train)
        pred = model.predict(data_model.X_test)
        print('Accuracy of {}:'.format(name), accuracy_score(pred, data_model.y_test) * 100)

        mat = confusion_matrix(data_model.y_test, pred)
        plot_confusion_matrix(conf_mat=mat, figsize=(15, 15), show_normed=True)
        filename = "output/" + name.lower() + "_model_confusion_matrix.png"
        plt.savefig(filename, dpi=100)
