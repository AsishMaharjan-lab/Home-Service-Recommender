# ml_logic/train.py
import pandas as pd
import joblib
import os
import sys

# Adjust path to import from ml_logic.preprocess
# This ensures it works when run from app.py or standalone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_logic.preprocess import load_and_preprocess_data

def train_and_save_model(data_path="data/service_dataset.csv", model_dir="models"):
    """
    Loads data, preprocesses it, and saves the preprocessed data (feature matrix)
    and the fitted encoders/vectorizers.

    Args:
        data_path (str): Path to the raw dataset.
        model_dir (str): Directory to save the processed data and encoders.
    """
    print("Starting model training (data preparation and saving)...")

    # Ensure model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load and preprocess data
    processed_df, encoders = load_and_preprocess_data(file_path=data_path)

    # Save the preprocessed DataFrame (which acts as our feature store for recommendations)
    processed_df_path = os.path.join(model_dir, "processed_data.pkl")
    joblib.dump(processed_df, processed_df_path)
    print(f"Preprocessed data saved to: {processed_df_path}")

    # Save the encoders/vectorizers
    encoders_path = os.path.join(model_dir, "encoders.pkl")
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saved to: {encoders_path}")

    print("Model training (data preparation) complete.")

if __name__ == '__main__':
    # This block is for testing the training script independently
    print("Running training test...")
    # Ensure 'data' directory exists and has the dataset for testing
    # Adjust path for standalone run
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataset_test_path = os.path.join(data_dir, 'service_dataset.csv')

    if not os.path.exists(dataset_test_path):
        # Create a dummy CSV if it doesn't exist for standalone testing
        dummy_data = {
            'ID': [1, 2, 3],
            'Name': ['Test Plumber', 'Test Electrician', 'Test Painter'],
            'Service Type': ['Plumber', 'Electrician', 'Painter'],
            'Skills': ['Pipe Repair, Leak Fix', 'Wiring, Light Fix', 'Wall Painting'],
            'Location': ['Kathmandu', 'Lalitpur', 'Bhaktapur'],
            'Rating': [4.5, 3.8, 4.2],
            'Days Available': ['Mon–Fri', 'Tue–Sat', 'Wed–Sun'],
            'Contact': [9801000000, 9802000000, 9803000000]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(dataset_test_path, index=False)
        print(f"Created a dummy '{dataset_test_path}' for testing.")

    # Adjust model_dir for standalone run
    model_test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    train_and_save_model(data_path=dataset_test_path, model_dir=model_test_dir)
