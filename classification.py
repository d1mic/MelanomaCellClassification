from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import graphviz
import numpy as np


def main():

    df = pd.read_csv('IntersectMelanoma.csv', index_col=0)

    #x_data = df.loc[:, df.columns != 'Class'].values
    #y_class = np.transpose(df.loc[:, ['Class']].values)[0]

    x_data = df.drop('Class', axis=1)
    y_class = df.Class

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_class, test_size=0.3)

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print('Train acc: {}'.format(clf.score(x_train, y_train)))
    print('Test acc: {}'.format(clf.score(x_test, y_test)))

    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=df.loc[:, df.columns != 'Class'].columns,
                                    class_names=['melanoma1', 'melanoma2'])
    graph = graphviz.Source(dot_data)
    graph.render("Melanoma")


if __name__ == '__main__':
    main()