from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

def all_models_test(models:dict, X, y):
    """Run X and y thru all models in models dict and get performance"""
    # Perform Leave-One-Out Cross Validation (LOOCV) for each model
    predictions = {}

    for model_name, model in models.items():
        loocv = LeaveOneOut()
        all_y = []
        all_y_pred = []
        
        for train_index, test_index in loocv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Make predictions on the test data
            y_pred = model.predict(X_test)
            
            all_y.append(*y_test)
            all_y_pred.append(*y_pred)
        
        # Calculate average MSE across all folds
        rmse = mean_squared_error(all_y, all_y_pred, squared=False)
        r2 = r2_score(all_y, all_y_pred)
        predictions[model_name] = {
            "y": all_y, 
            "y_pred": all_y_pred, 
            "r2": r2,
            "rmse": rmse 
            }

        print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2*100:.4f}%")
    return predictions


def plot_predicted_vs_actual(predictions, visualize_model):
    rf_y = predictions[visualize_model]['y']
    rf_y_pred = predictions[visualize_model]['y_pred']

    plt.figure(figsize=(5,4))
    plt.scatter(rf_y, rf_y_pred, color='blue', alpha=0.5)
    plt.plot(rf_y, rf_y, color='red', linestyle='--')  # Plot diagonal line for reference (y = x)

    # Customize plot labels and title
    plt.title(f'{visualize_model} Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend(['Ideal Predictions', 'Predictions'], loc='best')

    # Show plot
    plt.show()


def run_binary_classification_models(X, y):
    """Function that takes X and y as input and passes it 
    through standard binary classification algorithms and returns performance 
    using Leave-One-Out cross-validation(ideal for small datasets). 
    Can be used for baseline assessment.    
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=55),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    predictions = run_classification_models(classifiers, X, y)
    return predictions

def run_multiclass_classification_models(X, y):
    """Function that takes X and y as input and passes it 
    through standard multiclass classification algorithms and returns performance 
    using Leave-One-Out cross-validation(ideal for small datasets). 
    Can be used for baseline assessment. 
    """
    classifiers = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(random_state=55),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(probability=True),
        # 'Multinomial Naive Bayes': MultinomialNB(),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
    }
    predictions = run_classification_models(classifiers, X, y)
    return predictions


def run_classification_models(classifiers, X, y):
    """Function that takes a dict of models to run X and y on and returns the results and predictions"""    
    # Initialize results dictionary
    predictions = {}
    
    # Perform LeaveOneOut cross-validation for each classifier
    for name, clf in classifiers.items():

        loo = LeaveOneOut()
        all_y = []
        all_y_pred = []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            all_y.append(*y_test)
            all_y_pred.append(*y_pred)

        accuracy = accuracy_score(all_y, all_y_pred)
        recall = recall_score(all_y, all_y_pred, average='macro')
        predictions[name] = {
            "y": all_y, 
            "y_pred": all_y_pred, 
            "accuracy": accuracy,
            "recall": recall 
            }
        
        print(f"{name} - Accuracy: {accuracy*100:.2f}%, Recall: {recall*100:.4f}%")
        # Store results
        predictions[name] = {"Accuracy": accuracy, "Recall": recall}
    
    return predictions


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
