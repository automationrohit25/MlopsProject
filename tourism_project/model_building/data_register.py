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

    # Define expected columns after cleaning (excluding CustomerID)
    expected_columns = [
        'Unnamed: 0', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]

    # Check if all expected columns are present (adjusting for Unnamed: 0 which might be an artifact)
    current_columns = df.columns.tolist()
    # Filter out 'Unnamed: 0' if it's not explicitly in the expected list and present in df
    if 'Unnamed: 0' in current_columns and 'Unnamed: 0' not in expected_columns:
        temp_df_cols = [col for col in current_columns if col != 'Unnamed: 0']
        temp_expected_cols = [col for col in expected_columns if col != 'Unnamed: 0']
    else:
        temp_df_cols = current_columns
        temp_expected_cols = expected_columns

    missing_columns = [col for col in temp_expected_cols if col not in temp_df_cols]
    if missing_columns:
        print(f"Error: Missing expected columns after cleaning: {missing_columns}")
        # Decide if you want to stop or continue with a warning
    else:
        print("All expected columns are present (after accounting for 'Unnamed: 0').")

    print("\n--- Dataset Summary ---")
    print(df.info())
    print("\n--- First 5 rows of cleaned data ---")
    print(df.head())
    print("\n--- Value counts for Gender (after cleaning) ---")
    print(df['Gender'].value_counts())
    print("\n--- Value counts for MaritalStatus (after cleaning) ---")
    print(df['MaritalStatus'].value_counts())

    # Save the cleaned DataFrame back to the original file path, overwriting it
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'tourism_project', 'data', 'tourism.csv')

    # If running from model_building directory in GitHub Actions
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'data', 'tourism.csv')

    register_data(data_path)
