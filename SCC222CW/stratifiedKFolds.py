import numpy as np
import pandas as pd
def stratified_k_folds(X, y, k=5, random_seed=39551261): 
    '''
    divides training and testing splits into k equal folds while ensuring that
    the ratio of classes to each training or testing dataset is equal for cross validation
    '''
    np.random.seed(random_seed)
    X = np.array(X)  
    y = np.array(y)
    folds = []
    classes = np.unique(y)
    fold_indices = [[] for _ in range(k)] # stores indicies belonging to i-th fold

    for Class in classes:
        class_indices = np.where(y == Class)[0] # gets indicies of current class
        np.random.shuffle(class_indices) 

        subsets = np.array_split(class_indices, k) # splits class indicies into k equal splits

        for i in range(k):
            fold_indices[i].extend(subsets[i]) # appends each split to fold i

    # Convert each fold list to a numpy array and shuffle it
    for i in range(k):
        fold_indices[i] = np.array(fold_indices[i])

    # Build (train_index, test_index) pairs
    all_indices = np.arange(len(X))
    for i in range(k):
        test_index = fold_indices[i]
        train_index = np.setdiff1d(all_indices, test_index) # returns values in all_indicies which aren't in test_index
        folds.append((train_index, test_index))

    return folds

df = pd.read_csv("glassDataset.csv")
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
X = df[features]
y = df['Type of glass']
SKF = stratified_k_folds(X, y, k=5, random_seed=39551261)