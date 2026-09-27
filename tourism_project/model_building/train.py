import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import os

def train_model(data_dir, output_dir):
    print(f"Loading data from: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, 'Xtrain.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'Xtest.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'ytrain.csv')).squeeze() # .squeeze() to convert DataFrame to Series
    y_test = pd.read_csv(os.path.join(data_dir, 'ytest.csv')).squeeze()

    # Define preprocessing steps
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    )

    # Define the model pipeline
    model_pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    )

    # Define hyperparameters for tuning
    param_grid = {
        'xgbclassifier__n_estimators': [100, 200],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__max_depth': [3, 5]
    }

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Hyperparameter tuning complete.")
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nBest Model Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # MLflow tracking
    # Removed mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") for GitHub Actions
    mlflow.set_experiment("Tourism_Prediction_Model")

    with mlflow.start_run():
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")

        # Save the best model
        model_path = os.path.join(output_dir, 'best_model.joblib')
        joblib.dump(best_model, model_path)
        print(f"Best model saved to {model_path}")
        mlflow.log_artifact(model_path, "model")

    print("MLflow tracking complete.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    # data_dir is where the Xtrain.csv etc. are located
    data_input_directory = os.path.join(current_dir, 'tourism_project', 'model_building')
    # If running from GitHub Actions, the artifacts are downloaded to the current directory
    if not os.path.exists(os.path.join(data_input_directory, 'Xtrain.csv')):
        data_input_directory = current_dir

    # output_dir is where the best_model.joblib will be saved
    model_output_directory = os.path.join(current_dir, 'tourism_project', 'deployment')
    # If running from GitHub Actions, this might need to be adjusted if not already existing
    os.makedirs(model_output_directory, exist_ok=True)

    train_model(data_input_directory, model_output_directory)
