import pandas as pd
import os

def register_data(file_path):
    print(f"Attempting to register data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)

    # --- Data Cleaning as per user request ---
    # 1. Customer ID column is not required.
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Dropped 'CustomerID' column.")

    # 2. In Gender column, Fe Male is mentioned .. that should be Female.
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        print("Corrected 'Fe Male' to 'Female' in 'Gender' column.")

    # 3. In Marital Status column, Single and Unamarried can be clubbed in one
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
        print("Combined 'Single' and 'Unmarried' in 'MaritalStatus' column.")

    # Define expected columns after cleaning
    expected_columns = [
        'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation', 'Gender',
        'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
        'Designation', 'MonthlyIncome', 'PitchSatisfactionScore',
        'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'
    ]

    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing expected columns after cleaning: {missing_columns}")
        return
    else:
        print("All expected columns are present.")

    print("\n--- Dataset Summary ---")
    print(df.info())
    print("\n--- First 5 rows of cleaned data ---")
    print(df.head())
    print("\n--- Value counts for Gender (after cleaning) ---")
    print(df['Gender'].value_counts())
    print("\n--- Value counts for MaritalStatus (after cleaning) ---")
    print(df['MaritalStatus'].value_counts())


if __name__ == "__main__":
    # Assuming the script is run from the project root or model_building directory
    # Adjust path if necessary
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'tourism_project', 'data', 'tourism.csv')

    # If running from model_building directory in GitHub Actions
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'data', 'tourism.csv')

    register_data(data_path)
