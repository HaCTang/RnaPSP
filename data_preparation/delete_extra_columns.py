import os
import pandas as pd

def process_csv_files(directory):
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Identify columns to drop (those starting with 'level_' or 'Unnamed:')
                columns_to_drop = [col for col in df.columns if col.startswith('level_') or col.startswith('Unnamed:')]
                
                # Drop the identified columns
                df = df.drop(columns=columns_to_drop, errors='ignore')
                
                # Save the processed DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                print(f"Processed and saved: {file_path}")

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to your directory
    directory_path = 'your_directory_path'
    process_csv_files(directory_path)
