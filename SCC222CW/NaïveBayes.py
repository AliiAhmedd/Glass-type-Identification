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
features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ba'] 
X = df[features]
y = df['Type of glass']


def getGaussianP(mean, variance, x):
    try:
        # converting all inputs to float to avoid type errors
        mean = np.asarray(mean, dtype=np.float64)
        variance = np.asarray(variance, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        variance[variance < 1e-9] = 1e-9  # Replace 0 variance with a very small number

        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))
    
    except ValueError:
        return 1e-9  

def trainNaïveBayes(X_train, y_train):
    classStats = {}
    uniqueClasses = np.unique(y_train) # returns a list with unique values of y
    for y in uniqueClasses:
        X_new = X_train[y_train == y] # X_new holds instances where y_train equal to the class y 
        means = np.mean(X_new, axis=0) # gets mean
        variances = np.var(X_new, axis=0) # gets variance
        probX = len(X_new) / len(X_train) # gets prior probability

        classStats[y] = {"means": means, "variances": variances, "probX": probX}
    return classStats

def predict(X_test, classStats):
    predictions = []  # list for storing y_pred
    uniqueClasses = list(classStats.keys())
    
    # Ensure X_test is a 2D array or DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values  # Convert DataFrame to NumPy array
    
    # Iterate over each test instance
    for test_instance in X_test:
        probabilities = {}  # stores a class' probability as the value and the class itself as the key
        
        for y in uniqueClasses:
            # Unpack mean, variance, and probX
            mean = classStats[y]["means"]
            variance = classStats[y]["variances"]
            probX = classStats[y]["probX"]
        
            # Calculate likelihood using Gaussian probability
            likelihood = np.prod(getGaussianP(mean, variance, test_instance))  # P(X1,...Xn|y)
            if likelihood is None or likelihood == 0:  
                continue
            probabilities[y] = likelihood * probX  # = P(y|X1,...Xn)
        
        # Assign the class with the highest probability as the prediction
            pred = max(probabilities, key=probabilities.get)
        predictions.append(pred)
    
    # Ensure number of predictions == number of test instances
    assert len(predictions) == len(X_test), f"Prediction mismatch: expected {len(X_test)}, got {len(predictions)}"
    
    return np.array(predictions)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))  # Unique class labels
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def k_folds_cross_validatedNB(X, y, k=5):
    """
    performs k-stratified-fold cross-validation using the MultiClassSVM 
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
        trainedNB = trainNaïveBayes(X_train, y_train)
        end_train = time.time()
        training_time = end_train - start_train

        # Predict on test set
        start_test = time.time()
        y_pred = predict(X_test, trainedNB)
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

    avg_accuracy = np.mean(accuracies)
    avg_time = sum(times)/k
    print(f"\nAverage Accuracy across {k} folds: {avg_accuracy:.2f}%")
    print(f"\nAverage Time across {k} folds: {avg_time:.2f}seconds")

cm = df.corr()
print("Correlation Matrix:\n", cm)
nb = k_folds_cross_validatedNB(X, y, k=5)
'''
Normal Feature Set:
40.29%
Dimensiality Reduction effect on accuracy(Highest accuracy gains with removal of features based on correlation values):
Ca, K: 42.16%
'''