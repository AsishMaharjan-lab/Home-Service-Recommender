# ml_logic/recommend.py
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_processed_data_and_encoders(model_dir="models"):
    """
    Loads the preprocessed data (feature matrix) and encoders.

    Args:
        model_dir (str): Directory where the processed data and encoders are saved.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed DataFrame.
            - dict: A dictionary of fitted vectorizers/encoders.
    """
    processed_df_path = os.path.join(model_dir, "processed_data.pkl")
    encoders_path = os.path.join(model_dir, "encoders.pkl")

    if not os.path.exists(processed_df_path) or not os.path.exists(encoders_path):
        print(f"Error: Processed data or encoders not found in {model_dir}. Please run train.py first.")
        return None, None

    try:
        processed_df = joblib.load(processed_df_path)
        encoders = joblib.load(encoders_path)
        print("Processed data and encoders loaded successfully.")
        return processed_df, encoders
    except Exception as e:
        print(f"Error loading processed data or encoders: {e}")
        return None, None

def preprocess_user_input(user_input, encoders, feature_cols_in_processed_data):
    """
    Preprocesses user input using the same encoders/vectorizers used for the dataset.
    Ensures the output user feature vector has the same columns as the trained data.

    Args:
        user_input (dict): Dictionary with user preferences (e.g., 'service_type', 'location', 'skills').
        encoders (dict): Dictionary of fitted vectorizers/encoders.
        feature_cols_in_processed_data (list): List of feature column names from the trained data.

    Returns:
        pd.DataFrame: A DataFrame representing the preprocessed user input feature vector,
                      aligned with the trained data's feature columns.
                      Returns None if required encoders are missing or input is invalid.
    """
    # Extract user input, provide empty strings as default
    service_type = user_input.get('service_type', '')
    skills = user_input.get('skills', '')
    location = user_input.get('location', '')
    days_available = user_input.get('days_available', '')

    # Initialize empty DataFrames for each feature type
    service_type_df = pd.DataFrame()
    skills_df = pd.DataFrame()
    days_available_df = pd.DataFrame()
    location_dummies = pd.DataFrame()

    # Apply TF-IDF for 'Service Type'
    tfidf_service_type = encoders.get('tfidf_service_type')
    if tfidf_service_type:
        service_type_features = tfidf_service_type.transform([service_type])
        service_type_df = pd.DataFrame(service_type_features.toarray(), columns=tfidf_service_type.get_feature_names_out()).add_prefix('ServiceType_')

    # Apply MultiLabelBinarizer for 'Skills'
    mlb_skills = encoders.get('mlb_skills')
    if mlb_skills:
        skills_list = [s.strip() for s in skills.split(',') if s.strip()]
        skills_features = mlb_skills.transform([skills_list])
        skills_df = pd.DataFrame(skills_features, columns=mlb_skills.classes_).add_prefix('Skill_')

    # Apply MultiLabelBinarizer for 'Days Available'
    mlb_days_available = encoders.get('mlb_days_available')
    if mlb_days_available:
        # Split by '–' or ',' or just space if user types single days
        days_list = [d.strip() for d in days_available.replace('–', ',').split(',') if d.strip()]
        days_available_features = mlb_days_available.transform([days_list])
        days_available_df = pd.DataFrame(days_available_features, columns=mlb_days_available.classes_).add_prefix('Day_')

    # Apply One-Hot Encoding for 'Location'
    location_columns_trained = encoders.get('location_columns')
    if location_columns_trained:
        # Create a dummy row for the location, then one-hot encode it
        dummy_location_df = pd.DataFrame([{'Location': location}])
        # Use pd.get_dummies on the 'Location' column
        location_dummies_raw = pd.get_dummies(dummy_location_df['Location'], prefix='Location')
        # Reindex to ensure all columns from training are present, filling missing with 0
        location_dummies = location_dummies_raw.reindex(columns=location_columns_trained, fill_value=0)

    # Combine all preprocessed user features
    user_features_list = [service_type_df, skills_df, days_available_df, location_dummies]
    user_features_list = [df for df in user_features_list if not df.empty] # Filter out empty DFs

    if not user_features_list:
        print("Warning: No features could be generated for user input. Check encoders or input.")
        # Create an empty DataFrame with the correct columns to avoid errors later
        return pd.DataFrame(0, index=[0], columns=feature_cols_in_processed_data)

    # Concatenate the generated feature DataFrames
    user_features_combined = pd.concat(user_features_list, axis=1)

    # Align the user's feature vector columns with the features columns from the processed_data
    # This is critical to ensure the dot product/cosine similarity works correctly.
    user_features_aligned = user_features_combined.reindex(columns=feature_cols_in_processed_data, fill_value=0)

    return user_features_aligned


def get_recommendations(user_input, processed_data, encoders, top_n=5, sort_by='similarity', sort_order='desc'):
    """
    Generates service recommendations based on user input, with filtering and sorting.

    Args:
        user_input (dict): Dictionary with user preferences.
        processed_data (pd.DataFrame): The preprocessed dataset of service providers.
        encoders (dict): Dictionary of fitted vectorizers/encoders.
        top_n (int): Number of top recommendations to return.
        sort_by (str): Column to sort by ('similarity', 'Rating', 'Name').
        sort_order (str): 'asc' for ascending, 'desc' for descending.

    Returns:
        list: A list of dictionaries, each representing a recommended service.
    """
    if processed_data is None or encoders is None:
        print("Error: Processed data or encoders not loaded. Cannot generate recommendations.")
        return []

    # Define the base columns that are not features for similarity calculation
    base_cols = ['ID', 'Name', 'Service Type', 'Location', 'Rating', 'Skills', 'Days Available', 'Contact']
    feature_cols_in_processed_data = [col for col in processed_data.columns if col not in base_cols]

    # Preprocess user input, ensuring alignment with processed_data's features
    user_features_df = preprocess_user_input(user_input, encoders, feature_cols_in_processed_data)

    if user_features_df is None or user_features_df.empty:
        print("Error: Failed to preprocess user input for recommendation.")
        return []

    # Convert both to numpy arrays for cosine similarity calculation
    provider_features = processed_data[feature_cols_in_processed_data].fillna(0).values
    user_features = user_features_df.fillna(0).values

    # Calculate cosine similarity
    if user_features.ndim == 1:
        user_features = user_features.reshape(1, -1)

    if provider_features.shape[0] == 0:
        print("No providers to compare against after initial filtering/loading.")
        return []

    similarities = cosine_similarity(user_features, provider_features).flatten()

    temp_df = processed_data.copy()
    temp_df['similarity'] = similarities

    # --- Filtering ---
    # Filter by user's minimum rating preference
    min_rating = float(user_input.get('rating', 1.0))
    filtered_data = temp_df[temp_df['Rating'] >= min_rating]

    # Filter by user's preferred service type if provided
    user_service_type = user_input.get('service_type', '').lower()
    if user_service_type:
        filtered_data = filtered_data[filtered_data['Service Type'].str.lower() == user_service_type]

    # Filter by user's preferred location if provided
    user_location = user_input.get('location', '').lower()
    if user_location:
        filtered_data = filtered_data[filtered_data['Location'].str.lower() == user_location]

    # Filter by user's preferred days available (check for overlap)
    user_days_input = user_input.get('days_available', '').lower()
    if user_days_input:
        user_days_list = []
        if '–' in user_days_input: # Handle ranges like Mon–Fri
            start_day_str, end_day_str = user_days_input.split('–')
            days_map = {'mon':0, 'tue':1, 'wed':2, 'thu':3, 'fri':4, 'sat':5, 'sun':6}
            reverse_days_map = {v: k for k, v in days_map.items()}
            start_idx = days_map.get(start_day_str[:3].lower())
            end_idx = days_map.get(end_day_str[:3].lower())
            if start_idx is not None and end_idx is not None:
                current_idx = start_idx
                while True:
                    user_days_list.append(reverse_days_map[current_idx])
                    if current_idx == end_idx:
                        break
                    current_idx = (current_idx + 1) % 7
        else: # Handle comma-separated or single days
            user_days_list = [d.strip()[:3].lower() for d in user_days_input.replace(',', ' ').split() if d.strip()]

        if user_days_list:
            def check_day_overlap(provider_days_str, user_days_norm):
                if not provider_days_str: return False
                provider_days_list = []
                if '–' in provider_days_str:
                    start_day_str, end_day_str = provider_days_str.split('–')
                    days_map = {'mon':0, 'tue':1, 'wed':2, 'thu':3, 'fri':4, 'sat':5, 'sun':6}
                    reverse_days_map = {v: k for k, v in days_map.items()}
                    start_idx = days_map.get(start_day_str[:3].lower())
                    end_idx = days_map.get(end_day_str[:3].lower())
                    if start_idx is not None and end_idx is not None:
                        current_idx = start_idx
                        while True:
                            provider_days_list.append(reverse_days_map[current_idx])
                            if current_idx == end_idx:
                                break
                            current_idx = (current_idx + 1) % 7
                else:
                    provider_days_list = [d.strip()[:3].lower() for d in provider_days_str.replace(',', ' ').split() if d.strip()]

                return any(day in user_days_norm for day in provider_days_list)

            filtered_data = filtered_data[filtered_data['Days Available'].apply(lambda x: check_day_overlap(str(x), user_days_list))]

    # --- Sorting ---
    if not filtered_data.empty:
        ascending = True if sort_order == 'asc' else False
        if sort_by == 'Rating':
            recommended_services = filtered_data.sort_values(by='Rating', ascending=ascending)
        elif sort_by == 'Name':
            recommended_services = filtered_data.sort_values(by='Name', ascending=ascending)
        else: # Default to similarity
            recommended_services = filtered_data.sort_values(by='similarity', ascending=False)
    else:
        recommended_services = pd.DataFrame() # Empty DataFrame if no matches after filtering

    # Return top N recommendations
    display_cols = ['ID', 'Name', 'Service Type', 'Location', 'Rating', 'Skills', 'Days Available', 'Contact']
    final_recommendations = []
    for index, row in recommended_services.head(top_n).iterrows():
        rec = {col: row[col] for col in display_cols if col in row}
        final_recommendations.append(rec)

    return final_recommendations

if __name__ == '__main__':
    print("Running recommendation test...")
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' not found. Please run train.py first to generate models.")
    else:
        processed_data, encoders = load_processed_data_and_encoders(model_dir)
        if processed_data is not None and encoders is not None:
            test_user_input = {
                'service_type': 'Plumber',
                'location': 'Kathmandu',
                'skills': 'Pipe Repair',
                'rating': '4.0',
                'days_available': 'Mon-Fri'
            }
            print("\nTest Recommendations (default sort):")
            recommendations = get_recommendations(test_user_input, processed_data, encoders, top_n=3)
            if recommendations:
                for rec in recommendations:
                    print(rec)
            else:
                print("No recommendations found for test input.")

            print("\nTest Recommendations (sort by Rating desc):")
            recommendations_rating = get_recommendations(test_user_input, processed_data, encoders, top_n=3, sort_by='Rating', sort_order='desc')
            if recommendations_rating:
                for rec in recommendations_rating:
                    print(rec)
            else:
                print("No recommendations found for test input with rating sort.")
        else:
            print("Skipping recommendation test due to missing data/encoders.")
