import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
import graphviz


"""
Finds zero the columns to remove depending on removal type (union or intersection)
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
Save the current df to the csv file
"""
def save_csv(df, filename):
    df.to_csv(filename + '.csv', sep=',')

"""
Seperate the data from df into data without class and class
"""
def seperate_data_class(df):
    v_data = df.values[:, :-1]
    v_class = df.values[:, -1:]
    return v_data, v_class

def decisionTree(x_train, x_test, y_train, y_test, criteria, depth=None, generateGraph=[]):
    clf = tree.DecisionTreeClassifier(criterion=criteria, max_depth=depth)
    clf.fit(x_train,y_train.ravel())

    print("---------------------------------------------\n")
    print("Rezultati za stablo: [" + str(criteria) + " " + str(depth) + "] \n")

    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))

    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))


    if generateGraph:
        dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names= generateGraph,
                                    class_names=['parental (BRAF inhibitor sensitive)', 'BRAF inhibitor resistant'])
        graph = graphviz.Source(dot_data)
        if(depth == None):
            depth = "full"
        graph.render("dtGraph/DecisionTree_" + criteria + "_" + str(depth))

def knn(x_train, x_test, y_train, y_test, n_neigh, weights='uniform', algorithm = 'auto'):
    clf = KNeighborsClassifier(n_neigh, weights, algorithm)
    clf.fit(x_train,y_train.ravel())

    print("---------------------------------------------\n")
    print("Rezultati za knn: [" + str(n_neigh) + " " + str(weights) + " " + str(algorithm) + "] \n")

    print('Trening tacnost: {}'.format(clf.score(x_train, y_train)))
    print('Test tacnost: {}'.format(clf.score(x_test, y_test)))

    y_predict_train = clf.predict(x_train)
    y_predict_test = clf.predict(x_test)
    print("Matrica kofuzije trening vrednosti: \n" + str(confusion_matrix(y_train, y_predict_train)))
    print("Matrica kofuzije test vrednosti: \n" + str(confusion_matrix(y_test, y_predict_test)))

    

def find_outlier_columns(df):
    outlier_res = (df.drop(columns="Class")).T
    outliers = outlier_res[(np.abs(stats.zscore(outlier_res)) >= 3).any(axis=1)]
    outlier_columns = list(outliers.T.columns)
    outlier_columns.append('Class')
    return df[outlier_columns]

def checkNaNvalues(df):
    if df.isnull().values.any():
        print("NaN vrednosti postoje\n")
    else:
        print("NaN vrednosti ne postoje\n")

def main():

    # Citanje prve kolone kao imena kolona
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

    # Joining 2 matrixes
    result_df = pd.concat([first_df, second_df], ignore_index=True)
    print("Dimenzija spojene matrice: " + str(result_df.shape) + "\n")
    
    # Checking NaN values
    checkNaNvalues(result_df)

    # Droping zeros
    result_df.drop(columns=columns_to_remove, inplace=True)
    print("Dimenzija posle izbacenih nula: " + str(result_df.shape) + "\n")

    # Finding outlier columns and setting them up for analisys
    outliers_df = find_outlier_columns(result_df)
    print("Dimenzija tabele autlajera: " + str(outliers_df.shape) + "\n")

    # Mixing everything
    result_df = result_df.sample(frac=1).reset_index(drop=True)
    outliers_df = outliers_df.sample(frac=1).reset_index(drop=True)

    # Getting attribute names for graph
    columns_without_class = result_df.loc[:, result_df.columns != 'Class'].columns.tolist()
    outlier_columns_without_class = outliers_df.loc[:, outliers_df.columns != 'Class'].columns.tolist()

    
    # Separete data and class values
    x,y = seperate_data_class(result_df)
    out_x, out_y = seperate_data_class(outliers_df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    ox_train, ox_test, oy_train, oy_test = train_test_split(out_x, out_y, test_size=0.4)

    # Decision tree testing
    #decisionTree(x_train, x_test, y_train, y_test, 'gini', None, columns_without_class)
    #decisionTree(x_train, x_test, y_train, y_test, 'entropy', None, columns_without_class)
    #decisionTree(x_train, x_test, y_train, y_test, 'gini', 6, columns_without_class)
    #decisionTree(x_train, x_test, y_train, y_test, 'entropy', 6, columns_without_class)
    #decisionTree(x_train, x_test, y_train, y_test, 'gini', 4, columns_without_class)
    #decisionTree(x_train, x_test, y_train, y_test, 'entropy', 4, columns_without_class)
    
    #decisionTree(ox_train, ox_test, oy_train, oy_test, 'gini', 4, outlier_columns_without_class)
    

    # Poredjenje algoritama
    #knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'kd_tree')
    #knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'ball_tree')
    #knn(x_train, x_test, y_train, y_test, 5, 'uniform')
    #knn(x_train, x_test, y_train, y_test, 5, 'uniform', 'brute')

    # Poredjenje broja suseda
    #knn(x_train, x_test, y_train, y_test, 3)
    #knn(x_train, x_test, y_train, y_test, 5)
    #knn(x_train, x_test, y_train, y_test, 10)
    #knn(x_train, x_test, y_train, y_test, 15)


    


if __name__ == "__main__":
    main()
