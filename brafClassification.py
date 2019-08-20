import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
import graphviz
import time


"""
Nalazi nula kolone i eliminise ih u zavisnosti od odabranog nacina - unija ili presek
"""
def zero_columns_to_remove(df1, df2, removal_type):
    first_zero_columns = df1.columns[(df1 == 0).all()]
    second_zero_columns = df2.columns[(df2 == 0).all()]

    if(removal_type == 'union'):
        removed_cols = np.union1d(first_zero_columns, second_zero_columns)
    elif (removal_type == 'intersection'):
        removed_cols = np.intersect1d(first_zero_columns, second_zero_columns)
    else:
        print("Bad type of removing zero values")

    return removed_cols

"""
Cuva trenutni CSV fajl
"""
def save_csv(df, filename):
    df.to_csv(filename + '.csv', sep=',')

"""
Razdvaja klasu od podataka i vraca oba
"""
def seperate_data_class(df):
    v_data = df.values[:, :-1]
    v_class = df.values[:, -1:]
    return v_data, v_class


"""
STABLO ODLUCIVANJA - klasifikacija
"""
def decisionTree(x_train, x_test, y_train, y_test, criteria, depth=None, generateGraph=[]):
    print("---------------------------------------------\n")
    print("Rezultati za stablo: [" + str(criteria) + " " + str(depth) + "] \n")
    beginTime = time.time()
    print("Kreiranje klasifikatora ... \n")
    clf = tree.DecisionTreeClassifier(criterion=criteria, max_depth=depth)
    clf.fit(x_train,y_train.ravel())
    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))
    endTime = time.time()
    elapsedTime = endTime - beginTime
    print(f"Vreme potrebno za izvrsavanje: {elapsedTime:.4f} \n")
    if generateGraph:
        print("Kreiranje grafa")
        dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names= generateGraph,
                                    class_names=['parental (BRAF inhibitor sensitive)', 'BRAF inhibitor resistant'])
        graph = graphviz.Source(dot_data)
        if(depth == None):
            depth = "full"
        graph.render("dtGraph/DecisionTree_" + criteria + "_" + str(depth))


"""
KNN - klasifikacija
"""
def knn(x_train, x_test, y_train, y_test, n_neigh, weights='uniform', algorithm = 'auto'):
    print("---------------------------------------------\n")
    print("Rezultati za knn: [" + str(n_neigh) + " " + str(weights) + " " + str(algorithm) + "] \n")
    beginTime = time.time()
    print("Kreiranje klasifikatora ... \n")
    clf = KNeighborsClassifier(n_neigh, weights, algorithm)
    clf.fit(x_train,y_train.ravel())
    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))
    endTime = time.time()
    elapsedTime = endTime - beginTime
    print(f"Vreme potrebno za izvrsavanje: {elapsedTime:.4f} \n")

"""
SVM - klasifikacija
"""
def svm(x_train, x_test, y_train, y_test, kernel, gamma='auto'):
    print("---------------------------------------------\n")
    print("Rezultati za svm: [" + str(kernel) + " " + str(gamma) + "] \n")
    beginTime = time.time()
    print("Kreiranje klasifikatora ... \n")
    clf = SVC(kernel=kernel, gamma=gamma)
    clf.fit(x_train,y_train.ravel())
    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))
    endTime = time.time()
    elapsedTime = endTime - beginTime
    print(f"Vreme potrebno za izvrsavanje: {elapsedTime:.4f} \n")

"""
BAGGING - klasifikacija
"""
def bagging(x_train, x_test, y_train, y_test, n, classifier=None):
    print("---------------------------------------------\n")
    print("Rezultati za bagging: [" + str(classifier) + " " + str(n) + "] \n")
    beginTime = time.time()
    print("Kreiranje klasifikatora ... \n")
    if(classifier == "svc"):
        unit = SVC(kernel='poly', gamma='auto')
    elif(classifier == "tree"):
        unit = tree.DecisionTreeClassifier()
    elif(classifier == "knn"):
        unit = KNeighborsClassifier(3, algorithm="brute")
    else:
        unit = None

    clf = BaggingClassifier(unit, n_estimators=n)
    clf.fit(x_train,y_train.ravel())
    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))
    endTime = time.time()
    elapsedTime = endTime - beginTime
    print(f"Vreme potrebno za izvrsavanje: {elapsedTime:.4f} \n")

"""
BOOSTING - klasifikacija
"""
def boosting(x_train, x_test, y_train, y_test, learning=1.0, classifier=None):
    print("---------------------------------------------\n")
    print("Rezultati za boosting: [" + str(classifier) + " " + str(learning) + "] \n")
    beginTime = time.time()
    print("Kreiranje klasifikatora ... \n")
    if(classifier == "svc"):
        unit = SVC(kernel='poly', gamma='auto')
        alg = "SAMME"
    elif(classifier == "tree"):
        unit = tree.DecisionTreeClassifier()
        alg = "SAMME.R"
    else:
        alg = "SAMME.R"
        unit = None
    
    clf = AdaBoostClassifier(unit, learning_rate=learning, algorithm = alg)
    clf.fit(x_train,y_train.ravel())
    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))
    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))
    endTime = time.time()
    elapsedTime = endTime - beginTime
    print(f"Vreme potrebno za izvrsavanje: {elapsedTime:.4f} \n")


"""
Nalazi kolone van granica
"""
def find_outlier_columns(df):
    outlier_res = (df.drop(columns="Class")).T
    outliers = outlier_res[(np.abs(stats.zscore(outlier_res)) >= 3).any(axis=1)]
    outlier_columns = list(outliers.T.columns)
    outlier_columns.append('Class')
    return df[outlier_columns]

"""
Proverava nedostajuce vrednosti
"""
def checkNaNvalues(df):
    if df.isnull().values.any():
        print("NaN vrednosti postoje\n")
    else:
        print("NaN vrednosti ne postoje\n")

def main():

    print("PRETRPOCESIRANJE")
    print("---------------------------------------------\n")
    print("Ucitavanje podataka ... \n")
    first_df = pd.read_csv('001_Melanoma_Cell_Line_csv.csv', index_col=0)
    second_df = pd.read_csv('002_Melanoma_Cell_Line_csv.csv', index_col=0)

    # Transponovanje matrice
    first_df = first_df.T
    second_df = second_df.T
    print("Pocetna dimenzija 1. matrice: " + str(first_df.shape))
    print("Pocetna dimenzija 2. matrice: " + str(second_df.shape) + '\n')
    
    # Dodavanje klase
    first_df.insert(len(first_df.columns), "Class", 'parental (BRAF inhibitor sensitive)')
    second_df.insert(len(second_df.columns), "Class", 'BRAF inhibitor resistant')
    print("Dimenzija 1. matrice posle dodavanja klase: " + str(first_df.shape))
    print("Dimenzija 2. matrice posle dodavanja klase: " + str(second_df.shape) + '\n')
    
    # Nalazenje kolona koje treba izbaciti na osnovu tipa (unija/presek)
    columns_to_remove = zero_columns_to_remove(first_df, second_df, removal_type="intersection")

    # Sjedinjavanje matrica
    result_df = pd.concat([first_df, second_df], ignore_index=True)
    print("Dimenzija spojene matrice: " + str(result_df.shape) + "\n")
    
    # Provera nedostajucih vrednosti
    checkNaNvalues(result_df)

    # Eliminacija nula
    result_df.drop(columns=columns_to_remove, inplace=True)
    print("Dimenzija posle izbacenih nula: " + str(result_df.shape) + "\n")

    # Pronalazenje autlajera
    outliers_df = find_outlier_columns(result_df)
    print("Dimenzija tabele autlajera: " + str(outliers_df.shape) + "\n")

    # Random promesati podatke
    result_df = result_df.sample(frac=1).reset_index(drop=True)
    outliers_df = outliers_df.sample(frac=1).reset_index(drop=True)

    # Uzimanje imena klasa za graf
    columns_without_class = result_df.loc[:, result_df.columns != 'Class'].columns.tolist()
    outlier_columns_without_class = outliers_df.loc[:, outliers_df.columns != 'Class'].columns.tolist()
 
    # Razdvajanje test i trening podataka
    x,y = seperate_data_class(result_df)
    out_x, out_y = seperate_data_class(outliers_df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    ox_train, ox_test, oy_train, oy_test = train_test_split(out_x, out_y, test_size=0.4)

    # TESTIRANJE STABLA ODLUCIVANJA - ceo skup
    '''
    decisionTree(x_train, x_test, y_train, y_test, 'gini', None)
    decisionTree(x_train, x_test, y_train, y_test, 'gini', 6 )
    decisionTree(x_train, x_test, y_train, y_test, 'gini', 4 )
    decisionTree(x_train, x_test, y_train, y_test, 'gini', 3 , columns_without_class)
    decisionTree(x_train, x_test, y_train, y_test, 'entropy', None)
    decisionTree(x_train, x_test, y_train, y_test, 'gini', 6 )
    decisionTree(x_train, x_test, y_train, y_test, 'entropy', 6, columns_without_class)
    decisionTree(x_train, x_test, y_train, y_test, 'gini', 4, columns_without_class)
    decisionTree(x_train, x_test, y_train, y_test, 'entropy', 4, columns_without_class)
    '''

    # TESTIRANJE KNN - ceo skup
    '''
    knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'kd_tree')
    knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'ball_tree')
    knn(x_train, x_test, y_train, y_test, 5, 'uniform')
    knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'brute')
    
    knn(x_train, x_test, y_train, y_test, 3, 'uniform', 'brute')
    knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'brute')
    knn(x_train, x_test, y_train, y_test, 10, 'uniform', 'brute')
    knn(x_train, x_test, y_train, y_test, 15, 'uniform', 'brute')
    '''

    # TESTIRANJE SVM-a - ceo skup

    '''
    svm(x_train, x_test, y_train, y_test, 'rbf')
    svm(x_train, x_test, y_train, y_test, 'poly')
    svm(x_train, x_test, y_train, y_test, 'sigmoid')
    '''

    # TESTIRANJE BAGGING - ceo skup

    '''
    bagging(x_train, x_test, y_train, y_test, 3)
    bagging(x_train, x_test, y_train, y_test, 5)
    bagging(x_train, x_test, y_train, y_test, 7)
    bagging(x_train, x_test, y_train, y_test, 10)

    bagging(x_train, x_test, y_train, y_test, 5, 'svc')
    bagging(x_train, x_test, y_train, y_test, 5, 'tree')
    bagging(x_train, x_test, y_train, y_test, 5, 'knn')
    '''

    # TESTIRANJE BOOSTING - ceo skup

    '''
    boosting(x_train, x_test, y_train, y_test, 0.5)
    boosting(x_train, x_test, y_train, y_test, 0.7)
    boosting(x_train, x_test, y_train, y_test, 0.9)
    boosting(x_train, x_test, y_train, y_test, 1)
    boosting(x_train, x_test, y_train, y_test, 1.2)

    boosting(x_train, x_test, y_train, y_test, 1, 'svc')
    boosting(x_train, x_test, y_train, y_test, 1, 'tree')
    '''

    # TESTIRANJE redukovanog skupa

    decisionTree(ox_train, ox_test, oy_train, oy_test, 'gini', 4, outlier_columns_without_class)
    knn(ox_train, ox_test, oy_train, oy_test, 3, 'uniform', 'brute')
    svm(ox_train, ox_test, oy_train, oy_test, 'poly')
    bagging(ox_train, ox_test, oy_train, oy_test, 5, 'svc')
    boosting(ox_train, ox_test, oy_train, oy_test, 1)


if __name__ == "__main__":
    main()
