import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree


def main():
    data = pd.read_csv("WeatherConditions.csv", delimiter=',')
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X, y)

    fn = ["Outlook", "Humidity", "Windy"]
    cn = ["No", "Yes"]
    tree.export_graphviz(clf,
                         out_file="tree.dot",
                         feature_names=fn,
                         class_names=cn,
                         filled=True)


if __name__ == '__main__':
    main()
