from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def multiclass_classification(X, y, n_splits=5, binary_classification=True):
    """
    Perform multiclass classification using k-fold cross-validation.
    
    Parameters:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
        n_splits (int): Number of splits for k-fold cross-validation.
        
    Returns:
        dict: Dictionary containing accuracy and recall scores for each model.
    """
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(random_state=55),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Linear SVM': SVC(kernel="linear", probability=True),        
        # 'Multinomial Naive Bayes': MultinomialNB(),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
    }
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, random_state=55, shuffle=True)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Initialize dictionary to store results
    predictions = {}
    
    # Perform k-fold cross-validation for each model
    for name, model in models.items():
        all_y, all_y_pred = [], []
        
        # Perform k-fold cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Standardize the data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test_scaled)

            all_y += y_test.tolist()
            all_y_pred += y_pred.tolist()

        accuracy = accuracy_score(all_y, all_y_pred)
        if binary_classification:
            recall = recall_score(all_y, all_y_pred)
            precision = precision_score(all_y, all_y_pred)
            f1Score = f1_score(all_y, all_y_pred)
        else:
            recall = recall_score(all_y, all_y_pred, average='macro')
            precision = precision_score(all_y, all_y_pred, average='macro')
            f1Score = f1_score(all_y, all_y_pred, average='macro')

    
        # Store results
        predictions[name] = {"F1-Score": f1Score, "Recall": recall, "Accuracy": accuracy, "Precision": precision}
    
    return predictions





def plot_feature_importance_plot_using_rf(X, y):
    """Function to plot feature importance plot using Random Forest Classifier
    Inputs:
        X: preprocessed X (excluding scaling)
        y: target
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=55)

    # Train the classifier
    rf_classifier.fit(X_scaled, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Get feature names
    feature_names = X.columns  # Replace with actual feature names if available


    # Sort the feature importances and feature names
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_feature_names, sorted_feature_importances)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance Plot')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
