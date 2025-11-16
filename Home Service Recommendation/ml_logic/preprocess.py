import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os

def load_and_preprocess_data(file_path="data/service_dataset.csv"):
    """
    Loads the dataset, handles missing values, and preprocesses text/categorical features.

    Args:
        file_path (str): The path to the dataset CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed DataFrame.
            - dict: A dictionary of fitted vectorizers/encoders for inverse transformation or future use.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    # --- Data Cleaning and Type Conversion (if necessary) ---
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df.dropna(subset=['Rating'], inplace=True) # Drop rows where Rating couldn't be converted

    # --- Feature Engineering & Preprocessing ---
    # Fill NaNs with empty string before processing text/categorical columns
    df['Service Type'] = df['Service Type'].fillna('').astype(str)
    df['Skills'] = df['Skills'].fillna('').astype(str)
    df['Location'] = df['Location'].fillna('').astype(str)
    df['Days Available'] = df['Days Available'].fillna('').astype(str)

    # For 'Skills' and 'Days Available' which are comma-separated, let's split them
    df['Skills_list'] = df['Skills'].apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
    # For days, handle ranges like 'Mon–Fri' by splitting on '–' or comma, then processing
    df['Days_Available_list'] = df['Days Available'].apply(lambda x: [d.strip() for d in x.replace('–', ',').split(',') if d.strip()])


    # Initialize encoders/vectorizers
    vectorizers = {}

    # TF-IDF for 'Service Type'
    tfidf_service_type = TfidfVectorizer(stop_words='english')
    service_type_features = tfidf_service_type.fit_transform(df['Service Type'])
    vectorizers['tfidf_service_type'] = tfidf_service_type


    # MultiLabelBinarizer for 'Skills_list'
    mlb_skills = MultiLabelBinarizer()
    skills_features = mlb_skills.fit_transform(df['Skills_list'])
    vectorizers['mlb_skills'] = mlb_skills


    # MultiLabelBinarizer for 'Days_Available_list'
    mlb_days_available = MultiLabelBinarizer()
    days_available_features = mlb_days_available.fit_transform(df['Days_Available_list'])
    vectorizers['mlb_days_available'] = mlb_days_available


    # One-Hot Encoding for 'Location'
    location_dummies = pd.get_dummies(df['Location'], prefix='Location')
    vectorizers['location_columns'] = location_dummies.columns.tolist() # Store columns for later consistency


    # Convert sparse matrices to DataFrame for concatenation with pandas dummies.
    service_type_df = pd.DataFrame(service_type_features.toarray(), columns=tfidf_service_type.get_feature_names_out(), index=df.index)
    skills_df = pd.DataFrame(skills_features, columns=mlb_skills.classes_, index=df.index)
    days_available_df = pd.DataFrame(days_available_features, columns=mlb_days_available.classes_, index=df.index)

    # Ensure all DataFrames have the same index before concatenating
    # IMPORTANT: Include original 'Service Type', 'Location', 'Skills', 'Days Available', 'Contact'
    # for display purposes in recommendations.
    processed_df = pd.concat([df[['ID', 'Name', 'Service Type', 'Location', 'Rating', 'Skills', 'Days Available', 'Contact']],
                              service_type_df.add_prefix('ServiceType_'),
                              skills_df.add_prefix('Skill_'),
                              days_available_df.add_prefix('Day_'),
                              location_dummies], axis=1)

    print("Data Preprocessing Complete. Shape of processed data:", processed_df.shape)
    print("Columns of processed data:", processed_df.columns.tolist())

    return processed_df, vectorizers

if __name__ == '__main__':
    # This block is for testing the preprocessing script independently
    print("Running preprocessing test...")
    # Ensure the 'data' directory exists for testing
    if not os.path.exists('data'):
        os.makedirs('data')
    # Create a dummy CSV for testing if the actual file isn't present during independent run
    if not os.path.exists('data/service_dataset.csv'):
        dummy_data = {
            'ID': [1, 2],
            'Name': ['Test Plumber', 'Test Electrician'],
            'Service Type': ['Plumber', 'Electrician'],
            'Skills': ['Pipe Repair, Leak Fix', 'Wiring, Light Fix'],
            'Location': ['Kathmandu', 'Lalitpur'],
            'Rating': [4.5, 3.8],
            'Days Available': ['Mon–Fri', 'Tue–Sat'],
            'Contact': [9801000000, 9802000000]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv('data/service_dataset.csv', index=False)
        print("Created a dummy 'service_dataset.csv' for testing.")

    processed_data, encoders = load_and_preprocess_data()
    print("\nProcessed Data Head:")
    print(processed_data.head())
    print("\nEncoders/Vectorizers collected:")
    print(encoders.keys())
