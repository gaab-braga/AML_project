def standardize_data(df, numeric_features):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler

    # Imputer for missing values
    imputer = SimpleImputer(strategy='median')
    df[numeric_features] = imputer.fit_transform(df[numeric_features])

    # Scaler for normalization
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df

def load_and_process_data(file_path):
    import pandas as pd

    # Load data
    df = pd.read_csv(file_path)

    # Define numeric features for standardization
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Apply standardization
    df = standardize_data(df, numeric_features)

    return df