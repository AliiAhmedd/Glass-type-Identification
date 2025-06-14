'''
main file used for running the algorithms
'''
Algorithm = int(input("Type 1 for KNN, 2 for Decision Tree, 3 for Naïve Bayes and 4 for SVM:"))
if Algorithm == 1:
    from KNN import knn
elif Algorithm == 2:
    from ID3DecisionTree import dt
elif Algorithm == 3:
    from NaïveBayes import nb
elif Algorithm == 4:
    from SVM import svm