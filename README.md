# News-Article-Classification

## Introduction
In this task, we study and implement different machine learning models on News Article Dataset to classify the articles into a set of categories (0-10).

## Dataset
The dataset is provided as part of a Kaggle competition. A split of 80%-20% is created for training and validation respectively.

## Preprocessing
In order to feed the input data (news article) as input to the machine learning models, the following preprocessing steps are followed:
* Tfidf Vectorization
* Stemming
* Lemmatization

## Models
Different models are studied and implemented for this task including
* Naive Bayes Classifier
* Multinomial NB Classifier
* Logistic Regression Classifier
* SGD Classifier
* Random Forest Classifier
* Adaboost Classifier
* XGBoost Classifier

## Handling data imbalance
The dataset provided in the competition is highly skewed and we incorporate sampling methods to ensure balance in the dataset. Some of the sampling methods studied and implemented include:
* Random oversampling
* Random undersampling
* SMOTE

## Hyperparameter Tuning
GridSearchCV is implemented on the dataset using 3-fold cross validation to determine the set of best parameters. 

## Results
The SGD classifier showed the best results with an accuracy of 87%. To handle imbalance, we utilised random oversampling.

