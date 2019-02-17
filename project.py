import pandas as pd
import numpy as np


def main():

    # Citanje prve kolone kao imena kolona
    first_df = pd.read_csv('001_Melanoma_Cell_Line_csv.csv', index_col=0)
    second_df = pd.read_csv('002_Melanoma_Cell_Line_csv.csv', index_col=0)

    # Transponovanje matrice
    first_df = first_df.T
    second_df = second_df.T
    
    # Dodavanje klase
    first_df.insert(len(first_df.columns), "Class", 'melanoma1')
    second_df.insert(len(second_df.columns), "Class", 'melanoma2')

    # Nalazenje 0 kolona
    first_zero_columns = first_df.columns[(first_df == 0).all()]
    second_zero_columns = second_df.columns[(second_df == 0).all()]

    # Nalazenje preseka i unije nula kolona
    intersection = np.intersect1d(first_zero_columns, second_zero_columns)
    union = np.union1d(first_zero_columns, second_zero_columns)


    # Izbacivanje nula kolona (opcija presek)
    intersect_df1 = first_df.drop(columns=intersection)
    intersect_df2 = second_df.drop(columns=intersection)
    intersect_result = pd.concat([intersect_df1, intersect_df2], ignore_index=True)
    intersect_result = intersect_result.sample(frac=1).reset_index(drop=True)


    # Izbacianje nula kolona (opcija unija)
    union_df1 = first_df.drop(columns=union)
    union_df2 = second_df.drop(columns=union)
    union_result = pd.concat([union_df1, union_df2], ignore_index=True)
    union_result = union_result.sample(frac=1).reset_index(drop=True)

    intersect_result.to_csv('IntersectMelanoma.csv', sep=',')
    union_result.to_csv('UnionMelanoma.csv', sep=',')


if __name__ == "__main__":
    main()