import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from stratifiedKFolds import SKF

class SVM:
    """
    linear SVM for binary classification with subgradient descent.
    uses hinge loss with regularization to updat weights and bias
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000, random_seed=39551261): # constructor
        self.learning_rate = learning_rate # controls the step size used when updating the weights and bias for subgradient descent
        self.lambda_param = lambda_param # penalizes large weight magnitudes to avoid overfitting
        self.iterations = iterations
        self.random_seed = random_seed
        self.w = None
        self.b = None

    def fit(self, X, y):
        np.random.seed(self.random_seed)  # ensures that calls to random functions gives the same sequence of values with each run
        rows, columns = X.shape
        # Random initialization for weights & bias
        self.w = np.random.randn(columns) # small random vector of length equal to the number of features
        self.b = np.random.randn()
        # initializing to random values instead of zeros is better as each weight and biar can now learn distinctly from the start

        for _ in range(self.iterations):
            for index, x in enumerate(X):
                if y[index] * (np.dot(x, self.w) - self.b) >= 1:
                    # update weight
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # update weight and bias
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x, y[index]))
                    self.b -= self.learning_rate * y[index]

    def getScore(self, X):
        return np.dot(X, self.w) - self.b # returns (w*x)-b for each sample, which is used to train the multi-class SVM using one VS rest


class MultiClassSVM: 
    '''
    Trains one SVM per class using the One VS Rest method
    One VS Rest:
    Trains one SVM per class k and the samples with class value k are labelled 1 while others -1
    For the multi class SVM to predict it generates a score for each class within each test instance
    and the class with the maximum score is then chosen
    '''
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000, random_seed=39551261):# constructor
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.random_seed = random_seed
        self.models = {}

    def runSVM(self, X, y):
        unique_classes = np.unique(y)
        self.models = {} # key = unique class, value = trained binary SVM model

        for Class in unique_classes:
            y_binary = np.where(y == Class, 1, -1) # converts the y column into binary values. if y is equal to Class sets it to 1 otherwise -1

            # trains a new SVM on these binary labels
            svm = SVM(
                learning_rate=self.learning_rate,
                lambda_param=self.lambda_param,
                iterations=self.iterations,
                random_seed=self.random_seed
            )
            svm.fit(X, y_binary)
            self.models[Class] = svm # each class has its own SVM

    def predict(self, X):
        y_pred = [] # stores predictions
        for x in X:
            classScores = {} # key = class, value = svm
            for c, model in self.models.items(): # model is of class SVM so it can call getScore
                score = model.getScore(x)
                classScores[c] = score

            best_class = max(classScores, key=classScores.get) # pick the class with the best score
            y_pred.append(best_class) # append this class to predictions list

        return np.array(y_pred)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))  # Unique class labels
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def k_folds_cross_validatedSVM(X, y, k=5):
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

        # Train MultiClassSVM
        start_train = time.time()
        MultiSVM = MultiClassSVM(
            learning_rate=0.001,
            lambda_param=0.01,
            iterations=1000,         
            random_seed=39551261       
        )
        MultiSVM.runSVM(X_train, y_train)
        end_train = time.time()
        training_time = end_train - start_train

        # Predict on test set
        start_test = time.time()
        y_pred = MultiSVM.predict(X_test)
        end_test = time.time()
        testing_time = end_test - start_test

        accuracy = np.mean(y_pred == y_test) * 100
        accuracies.append(accuracy)
        times.append(training_time+testing_time)

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

def main():
    df = pd.read_csv("glassDataset.csv")
    features = ['RI', 'Na', 'Mg', 'Al', 'K', 'Ba']
    X = df[features]
    y = df['Type of glass']

    cm = df.corr()
    print("Correlation Matrix:\n", cm)
    k_folds_cross_validatedSVM(X, y, k=5)

svm = main()
'''
Normal feature set: 56.65%
Dimensiality Reduction effect on accuracy:
Fe, Ca and Si: 59.96%
'''