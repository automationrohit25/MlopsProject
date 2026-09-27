import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(input_file_path, output_dir):
    print(f"Loading data from: {input_file_path}")
    df = pd.read_csv(input_file_path)

    # Drop 'Unnamed: 0' column if it exists (often an artifact of saving/loading CSVs)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Dropped 'Unnamed: 0' column.")

    # Define features (X) and target (y)
    # CustomerID should have been dropped by data_register.py, so it won't be here.
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the splits to CSV files
    X_train.to_csv(os.path.join(output_dir, 'Xtrain.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'Xtest.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'ytrain.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'ytest.csv'), index=False)

    print(f"Data preparation complete. Splits saved to {output_dir}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

if __name__ == "__main__":
    # Adjust paths for execution context
    current_dir = os.getcwd()
    data_input_path = os.path.join(current_dir, 'tourism_project', 'data', 'tourism.csv')

    # For GitHub Actions, data_register job ensures the data is in tourism_project/data
    # No need for complex path logic here, as it will always be downloaded to the correct path

    output_directory = os.path.join(current_dir, 'tourism_project', 'model_building')
    # If running from GitHub Actions, this might need to be adjusted if not already existing
    os.makedirs(output_directory, exist_ok=True)
    
    # Check if we are running in the context of GitHub Actions where artifacts might be in root
    if not os.path.exists(os.path.join(output_directory, 'Xtrain.csv')) and not os.path.exists(os.path.join(output_directory, 'Xtest.csv')):
        # This handles the case where prep.py is run directly and needs to create its own output dir
        # or in GitHub Actions where data splits are uploaded from root
        pass # output_directory is fine as tourism_project/model_building

    # When run in GitHub Actions, prep.py might be in the root of the runner's workspace, 
    # but the output for splits is expected in the model_building directory of the repo structure.
    # The artifact download ensures tourism_project/data/tourism.csv is present.
    # The splits should be saved relative to the current working directory, 
    # which is the root of the cloned repo in Actions.
    if os.path.basename(current_dir) == 'MlopsProject' and not os.path.exists(os.path.join(current_dir, 'tourism_project', 'model_building')):
        output_directory = current_dir # save in current directory if in actions workflow
    
    prepare_data(data_input_path, output_directory)
