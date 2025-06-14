import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from stratifiedKFolds import SKF

# Load dataset
df = pd.read_csv("glassDataset.csv")

# Feature columns and target
features = ['RI', 'Na', 'Mg', 'Al', 'K', 'Ba'] 
X = df[features]
y = df['Type of glass']

def Entropy(y):
    y = y.dropna() # k folds creates NaN values
    labels = set(y.values) # stores unique labels
    length = len(y) # stores number of samples
    freq = {} # stores frequency for each label
    probabilities = {} # stores probabilities for each label
    for i in list(labels):
        freq[i] = 0

    for label in y:
        freq[label] += 1

    for key, value in freq.items():
        probabilities[key] = value / length

    entropy = 0
    for key, value in probabilities.items():
        if value > 0:  # Avoid log2(0) as it is undefined
            entropy += value * np.log2(value)
    finalEntropy = -entropy
    return finalEntropy

# Feature entropy function
def featureEntropy(feature, optimalThreshold, glassDataSetClassified, X_new):
    length = len(y) # total number of samples
    weightedAverage = 0

    # Create a temporary copy for entropy calculation
    temp_feature = X_new[feature].astype(str).copy()  

    # Categorize the temporary copy
    for index, value in temp_feature.items():
        value = float(value)
        if value < optimalThreshold:
            temp_feature.at[index] = f'< {optimalThreshold}' # categorize as below the threshold
        else:
            temp_feature.at[index] = f'> {optimalThreshold}'
    # Calculate entropy using the temporary categorized column
    freqValues = {}  # key = unique value (e.g., < threshold or > threshold) and value = # of occurrences
    rows = {}  # key = unique value and value = all rows which include this unique value
    subsetEntropy = {}  # key = unique value and value = subsetEntropy
    subsetProbabilities = {}  # key = unique value and valueList = probability of each label's occurrence for a specific subset
    
    for value in temp_feature:  # Iterate through all values of the temporary categorized feature
        if value not in freqValues:
            freqValues[value] = 0
            rows[value] = []
        freqValues[value] += 1  # Count occurrences for weighted average
        rows[value].append(glassDataSetClassified[temp_feature == value])  # Stores all rows where the feature == value

    for value, subset in rows.items():
        subset = pd.concat(subset)  # Combine list of DataFrames into a single DataFrame
        subsetSize = len(subset)
        subsetProbabilities[value] = {} # probabilities for current subset

        target_labels = y[subset.index]  # Get corresponding target labels for the subset rows

        # calculates probabilities for each label in the subset
        for label in y.unique():
            subsetProbabilities[value][label] = (target_labels == label).sum() / subsetSize

    # calculates entropy for each subset
    for key, probabilitiesList in subsetProbabilities.items():
        subsetEntropy[key] = 0
        for value in probabilitiesList.values():
            if value > 0:  # if log2(0) just continue
                subsetEntropy[key] += value * np.log2(value)

    for key in subsetEntropy:
        subsetEntropy[key] = -subsetEntropy[key]

    # calculates weighted average 
    for key, entropy in subsetEntropy.items():
        weightedAverage += (freqValues[key] / length) * entropy
    return weightedAverage, optimalThreshold, feature

# Entropy-based binning for continuous valued features
def optimal_thresholdValue_search(feature, E):
    sorted_values = sorted(X[feature].unique()) # sorts unique values of a given feature
    optimalThreshold = None
    max_info_gain = -float('inf')

    for i in range(1, len(sorted_values)):
        threshold = (sorted_values[i - 1] + sorted_values[i]) / 2 # calculates average of two values
        left_split = y[X[feature] <= threshold]
        right_split = y[X[feature] > threshold]

        entropy_left = Entropy(left_split) # calc entropy of left subset
        entropy_right = Entropy(right_split) # calc entropy of right subset
        weightedAverage = (len(left_split) / len(y)) * entropy_left + (len(right_split) / len(y)) * entropy_right
        infoGain = E - weightedAverage

        # Update the optimal threshold
        if infoGain > max_info_gain:
            max_info_gain = infoGain
            optimalThreshold = threshold

    return optimalThreshold

# ID3 algorithm
def ID3(E, glassDataSetClassified, X_new, remaining_features, labelName):
    # if all examples have the same label, return leaf node
    uniqueClasses = glassDataSetClassified[labelName].unique()
    if len(uniqueClasses) == 1:
        return {'prediction': uniqueClasses[0]}

    # return majority_class if no features are left
    if not remaining_features:
        majority_class = glassDataSetClassified[labelName].value_counts().idxmax()
        return {'prediction': majority_class}

    InfoGain = list() # stores info gain for each feature
    thresholds = list() # stores optimal thresholds for each feature

    # Information Gain for all features
    for feature in remaining_features:
        optimalThreshold = optimal_thresholdValue_search(feature, E)
        thresholds.append(optimalThreshold)

        # Entropy using a TEMPORARY CATEGORIZED COPY
        feature_Entropy, _, _ = featureEntropy(feature, optimalThreshold, glassDataSetClassified, X_new)
        InfoGain.append(E - feature_Entropy)

    # Determine the best feature and threshold to split on
    maxInfoGain = max(InfoGain)
    best_feature_index = InfoGain.index(maxInfoGain)
    best_feature = list(remaining_features)[best_feature_index]
    threshold = thresholds[best_feature_index]

    # Split the data based on best feature's threshold value
    left_split = glassDataSetClassified[glassDataSetClassified[best_feature] <= threshold]
    right_split = glassDataSetClassified[glassDataSetClassified[best_feature] > threshold]

    # returns the prediction if no further splitting is possible
    if len(left_split) == 0 or len(right_split) == 0:
        majority_class = glassDataSetClassified[labelName].value_counts().idxmax()
        return {'prediction': majority_class}

    remaining_features = remaining_features - {best_feature}  # Remove used feature

    # calculates entropy for left and right subtrees
    E_left = Entropy(left_split[labelName])
    E_right = Entropy(right_split[labelName])

    # recursivly call lwft and right branches
    left_branch = ID3(E_left, left_split, left_split[features], remaining_features, labelName)
    right_branch = ID3(E_right, right_split, right_split[features], remaining_features, labelName)

    # return current node
    return {
        'feature': best_feature,
        'threshold': threshold,
        'left': left_branch,
        'right': right_branch,
        'prediction': None
    }

# Decision tree function
def decisionTree(X_train, y_train, features):
    X_train = pd.DataFrame(X_train, columns=features) # create a pandas dataframe out of X_train
    glassDataSetClassified = X_train.copy()  # Copy of glass_dataset that deals with the decision tree
    remaining_features = set(features)  # Track remaining features
    labelName = 'Type of glass' 
    glassDataSetClassified[labelName] = pd.Series(y_train).reset_index(drop=True)

    # Get overall entropy or entropy of labels
    E = Entropy(glassDataSetClassified[labelName])

    tree = ID3(E, glassDataSetClassified, X_train, remaining_features, labelName)
    if tree is None:
        raise ValueError("Error: ID3() returned None, tree was not built correctly!")

    # Display the tree
    return tree


# Predict function
def predict(tree, sample):
    # Base case
    if tree.get('prediction') is not None:
        return tree['prediction']

    feature = tree['feature']
    threshold = tree['threshold']

    # Determine whether to go left or right
    if sample[feature] <= threshold:
        return predict(tree['left'], sample)
    else:
        return predict(tree['right'], sample)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))  # Unique class labels
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def k_folds_cross_validatedDT(X, y, k=5):
    """
    performs k-stratified-folded cross-validation on the dataset using KNN 
    """
    folds = SKF
    accuracies = []
    times = []

    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"\nFold {i+1}")

        # Split data based on the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] # selects training and testing rows from X
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] # selects training and testing rows from y 

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        start_train = time.time()
        trained_tree = decisionTree(X_train, y_train, features)
        end_train = time.time()
        training_time = end_train - start_train

        # Make Predictions & Measure Testing Time
        X_test = pd.DataFrame(X_test, columns=features)  # Ensure X_test is a pandas DataFrame
        start_test = time.time()
        y_pred = [predict(trained_tree, X_test.iloc[i]) for i in range(len(X_test))]
        end_test = time.time()
        testing_time = end_test - start_test

        # Compute accuracy
        accuracy = np.mean(y_pred == y_test) * 100
        accuracies.append(accuracy)
        times.append(training_time+testing_time)

        # Print results
        print("True Labels:   ", y_test.values)
        print("Predicted Labels:", y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.2f}%")
        print(f"Training Time: {training_time}\nTesting Time: {testing_time}")

        print(plot_confusion_matrix(y_test, y_pred))
        

    # Compute the average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    avg_time = sum(times)/k
    print(f"\nAverage Accuracy across {k} folds: {avg_accuracy:.2f}%")
    print(f"\nAverage Time across {k} folds: {avg_time:.2f}seconds")

cm = df.corr()
print("Correlation Matrix:\n", cm)
dt = k_folds_cross_validatedDT(X, y, k=5)
'''
Normal feature set: 44.91%
Dimensiality Reduction effect on accuracy(Highest accuracy gains with removal of features based on correlation values):
Ca, K, Fe and Si: 47.62%
Average Training + Testing Time:
1.80s
'''