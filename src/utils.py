import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    """Save a Python object using pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    return file_path



def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Train and evaluate multiple models, returning their performance scores."""
    model_report = {}
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        model_report[model_name] = accuracy
    
    return model_report
