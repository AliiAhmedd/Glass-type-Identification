import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2)) # p1 and p2 are arrays representing all feature values

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))  # Sum of absolute differences

def cosine_distance(p1, p2):
    dot_product = np.dot(p1, p2)  # Dot product of p1 and p2
    norm_p1 = np.linalg.norm(p1)  # Euclidean norm (magnitude) of p1
    norm_p2 = np.linalg.norm(p2)  # Euclidean norm (magnitude) of p2
    return 1 - (dot_product / (norm_p1 * norm_p2))  # Cosine distance = 1 - cosine similarity

def get_neighbors(X_train, y_train, test_instance, k=1):
    y_train = np.array(y_train)
    distances = [(manhattan_distance(test_instance, X_train[i]), y_train[i]) for i in range(len(X_train))]# distances will now include a list of tuples where the 1st value is the euclidean distance between
    #the test_instance and X_train[i] and the second value just maps the distance to a label
    distances.sort(key=lambda x: x[0])  # distances is now sorted based on the first value of the tuples which is the euclidean distance to ensure that nearest neighbors are at the beginning of thelist
    return [label for _, label in distances[:k]] # returns first k label values of the sorted

def predict(X_train, y_train, X_test, k=1):
    # uses get_neighbors to find the KNN, then evaluates most_common(1) according to the frequencies calculated by Counter, [0][0] extracts the predicted label from the result of most_common(1)
    return np.array([Counter(get_neighbors(X_train, y_train, x, k)).most_common(1)[0][0] for x in X_test])

def compute_accuracy(y_test, y_pred):
    return np.mean(y_test == y_pred) * 100 # computes mean of this boolean array which includes True values when y_test == y_pred

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))  # Unique class labels
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def k_folds_cross_validatedKNN(X, y, k=5):
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

        # Predict on test set
        start_trainANDtest = time.time()
        y_pred = predict(X_train, y_train, X_test, k=3)
        end_trainANDtest = time.time()
        trainingANDtesting_time = end_trainANDtest-start_trainANDtest

        accuracy = np.mean(y_pred == y_test) * 100
        accuracies.append(accuracy)
        times.append(trainingANDtesting_time)

        print("True Labels:   ", y_test.values)
        print("Predicted Labels:", y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.2f}%")
        print(f'Training + Testing Time: {trainingANDtesting_time}')

        print(plot_confusion_matrix(y_test, y_pred))
        
    avg_accuracy = np.mean(accuracies)
    avg_time = sum(times)/k
    print(f"\nAverage Accuracy across {k} folds: {avg_accuracy:.2f}%")
    print(f"\nAverage Time across {k} folds: {avg_time:.2f}seconds")

cm = df.corr()
print("Correlation Matrix:\n", cm)
knn = k_folds_cross_validatedKNN(X, y, k=5)
'''
Normal feature set: 75.39%
Dimensiality Reduction effect on accuracy(Highest accuracy gains with removal of features based on correlation values):
Ca, Fe and Si: 75.31%
Average Training + Testing Time:
0.03s
Setting the number of nearest neighbors to be inspected, k = 5 yielded to the highest value
'''