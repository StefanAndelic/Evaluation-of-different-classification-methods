# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import sys

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


SEED = 309
random.seed(SEED)
np.random.seed(SEED)
train_test_split_test_size = 0.3

def load():
    
    #reads in the dataset    
    # uncomment to test the program through a command line 
    file1 = sys.argv[1]   
    file2 = sys.argv[2]   
    training_file = pd.read_csv(file1,header = None)
    test_file = pd.read_csv(file2, skiprows=[0], header = None)
    #print(file)
    
    #training_file = pd.read_csv('adult.data',header = None);
    #first line does not contain useful information so we have to skip
    #test_file = pd.read_csv('adult.test',header= None)
    #print(test_file)
    #print(training_file[14].value_counts())
    #Exploratory Data Analysis
    #prints the variables type of attributes
    #print(training_file.info())
    #print(test_file.info())
    #prints the total number of rows and columns
    #print(training_file.shape)
    #prints the skewness
    #skew = training_file.skew()
    #print(skew)
    #visulisation
    #corr = training_file.corr(method = 'pearson') # Correlation Matrix
    #sns.heatmap(training_file.corr(), cmap='coolwarm',annot = True)
    #plt.show()
    #print(training_file.isnull().sum())
    
    
    
    return training_file, test_file

def preprocess(training_file,test_file):
    training_file.replace(' ?',value= np.NaN)
    test_file.replace(' ?',value= np.NaN)
    
    training_file = training_file.apply(lambda x: x.fillna(x.value_counts().index[0]))
    test_file = test_file.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
#     training_file.drop(training_file[4],inplace = True)
#     test_file.drop(test_file[4],inplace = True)
    
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()

    training_file[1] = labelencoder.fit_transform(training_file[1])
    training_file[3] = labelencoder.fit_transform(training_file[3])
    training_file[5] = labelencoder.fit_transform(training_file[5])
    training_file[6] = labelencoder.fit_transform(training_file[6])
    training_file[7] = labelencoder.fit_transform(training_file[7])
    training_file[8] = labelencoder.fit_transform(training_file[8])
    training_file[9] = labelencoder.fit_transform(training_file[9])
    training_file[13] = labelencoder.fit_transform(training_file[13])
    training_file[14] = labelencoder.fit_transform(training_file[14])
    #print(file)

    test_file[1] = labelencoder.fit_transform(test_file[1])
    test_file[3] = labelencoder.fit_transform(test_file[3])
    test_file[5] = labelencoder.fit_transform(test_file[5])
    test_file[6] = labelencoder.fit_transform(test_file[6])
    test_file[7] = labelencoder.fit_transform(test_file[7])
    test_file[8] = labelencoder.fit_transform(test_file[8])
    test_file[9] = labelencoder.fit_transform(test_file[9])
    test_file[13] = labelencoder.fit_transform(test_file[13])
    test_file[14] = labelencoder.fit_transform(test_file[14])

    #corr = training_file.corr(method = 'pearson') # Correlation Matrix
    #sns.heatmap(training_file.corr(), cmap='coolwarm',annot = True)
    #plt.show()

      

    return training_file, test_file
    


def split(training_file, test_file):
    training_file_copy= training_file.copy()
    test_file_copy = test_file.copy()

    X_train = training_file_copy.iloc[:, 1:-1].values 
    Y_train = training_file_copy.iloc[:, 14].values

    X_test = test_file_copy.iloc[:, 1:-1].values 
    Y_test = test_file_copy.iloc[:, 14].values

    return X_train,Y_train,X_test,Y_test

def model(X_train_full, X_train_label, technique):
    
    if technique == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier()
        model = neigh.fit(X_train_full,X_train_label)
        return model
    elif technique == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        naivebayes = GaussianNB()
        model = naivebayes.fit(X_train_full,X_train_label)
    elif technique == "SVM":
        from sklearn.svm import SVC
        svm = SVC(gamma = 'scale', max_iter=10)
        model = svm.fit(X_train_full,X_train_label)
    elif technique =="DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier();
        model = tree.fit(X_train_full,X_train_label)
    elif technique == "RandomForrest" :
        from sklearn.ensemble import RandomForestClassifier
        random_forrest = RandomForestClassifier()
        model = random_forrest.fit(X_train_full,X_train_label)
    elif technique == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        ada_boost = AdaBoostClassifier()
        model = ada_boost.fit(X_train_full,X_train_label)
    elif technique == "LinearDiscriminant" :
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        linear_discriminant = LinearDiscriminantAnalysis()
        model = linear_discriminant.fit(X_train_full,X_train_label)
    elif technique == "GradientBoosting" :
        from sklearn.ensemble import GradientBoostingClassifier
        gradient_boosting = GradientBoostingClassifier()
        model = gradient_boosting.fit(X_train_full,X_train_label)
    elif technique == "MLP":
        from sklearn.neural_network import MLPClassifier
        MLP = MLPClassifier()
        model = MLP.fit(X_train_full,X_train_label)
    elif technique == "LogisticClassification":
        from sklearn.linear_model import LogisticRegression
        logistic_regression = LogisticRegression()
        model = logistic_regression.fit(X_train_full,X_train_label)
    else:
        print("Classification Technique not found:"+ technique)


    return model



if __name__ == '__main__':
    file1, file2  = load()
    #print(file1)
    #print(file1.isnull().sum())
    #print(file2.isnull().sum())
    #preprocesses the file
    training, test = preprocess(file1, file2)
    
    #print(training)
    #print(test)
    #forms the X_train,Y_train,X_test,Y_test files
    X_train,Y_train,X_test,Y_test = split(training,test)
    #print(X_train)
    #print(Y_train)
    
    technique ="KNN";
    print("KNN classification:")
    knn_classification = model(X_train,Y_train,technique)
    y_pred = knn_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    technique ="NaiveBayes";
    print("Naive Bayes classification")
    naivebayes_classification = model(X_train,Y_train,technique)
    y_pred = naivebayes_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    technique="SVM"
    print("SVM Classification")
    svm_classification = model(X_train,Y_train,technique)
    y_pred = svm_classification.predict(X_test)
    print("Accuracy:", (metrics.accuracy_score(Y_test,y_pred)))
    print("F1 score:", (metrics.f1_score(Y_test,y_pred)))
    print("Precision score:", (metrics.precision_score(Y_test,y_pred)))
    print("ROC curve:", (metrics.roc_auc_score(Y_test,y_pred)))
    print("Recall:", (metrics.recall_score(Y_test,y_pred)))
    print('-----------------------------------------')
    technique="DecisionTree"
    print("Decision Tree classification")
    decisiontree_classification = model(X_train,Y_train,technique)
    y_pred = decisiontree_classification.predict(X_test)
    print("Accuracy:", (metrics.accuracy_score(Y_test,y_pred)))
    print("F1 score:", (metrics.f1_score(Y_test,y_pred)))
    print("Precision score:", (metrics.precision_score(Y_test,y_pred)))
    print("ROC curve:", (metrics.roc_auc_score(Y_test,y_pred)))
    print("Recall:", (metrics.recall_score(Y_test,y_pred)))
    print('-----------------------------------------')
    technique="RandomForrest"
    print("Random Forrest classification:")
    randomforrest_classification = model(X_train,Y_train,technique)  
    y_pred = randomforrest_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')   
    technique="AdaBoost"
    print("Ada Boost classification")
    adaboost_classification = model(X_train,Y_train,technique)  
    y_pred =  adaboost_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')   
    technique="GradientBoosting"
    print("Gradient Boosting classification")
    gradientboosting_classification = model(X_train,Y_train,technique)  
    y_pred = gradientboosting_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    technique="LinearDiscriminant"
    print("Linear Discriminant Analysis classification")
    lda_classification = model(X_train,Y_train,technique)  
    y_pred = lda_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    technique = "MLP"
    print("Multi Layer Perceptron classification")
    mlp_classification = model(X_train,Y_train,technique)  
    y_pred = mlp_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    technique = "LogisticClassification"
    print("Logistic classification")
    logistic_classification = model(X_train,Y_train,technique)  
    y_pred = logistic_classification.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(Y_test,y_pred),2))
    print("F1 score:", round(metrics.f1_score(Y_test,y_pred),2))
    print("Precision score:", round(metrics.precision_score(Y_test,y_pred),2))
    print("ROC curve:", round(metrics.roc_auc_score(Y_test,y_pred),2))
    print("Recall:", round(metrics.recall_score(Y_test,y_pred),2))
    print('-----------------------------------------')
    
