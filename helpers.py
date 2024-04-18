from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yaml

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




def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
