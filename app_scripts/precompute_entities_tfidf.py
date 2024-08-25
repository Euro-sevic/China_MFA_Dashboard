import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Function to split and clean the text data
def split_and_clean(text):
    return [t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"]

# Load the data
def load_data():
    print("Loading data...")
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', 'CMFA_PressCon_v4.xlsx')
    columns_to_load = ["id", "a_loc", "a_per", "a_org", "a_misc"]
    df = pd.read_excel(file_path, usecols=columns_to_load)
    
    print("Cleaning data...")
    columns_to_clean = ["a_loc", "a_per", "a_org", "a_misc"]
    for column in columns_to_clean:
        df[column] = df[column].replace("-", pd.NA).astype(str)
    return df

# Function to compute TF-IDF
def compute_tfidf_for_column(df, column_name):
    print(f"Computing TF-IDF for {column_name}...")
    tfidf_vectorizer = TfidfVectorizer(tokenizer=split_and_clean)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name].dropna())
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Save the TF-IDF data to a pickle file
def save_tfidf_data(tfidf_data, file_name):
    print(f"Saving TF-IDF data to {file_name}...")
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(tfidf_data, file)
    print(f"TF-IDF data saved to {file_path}")

# Main function
def main():
    df = load_data()

    # Compute TF-IDF for each entity column
    entity_columns = ["a_loc", "a_per", "a_org", "a_misc"]
    for column in entity_columns:
        tfidf_matrix, feature_names = compute_tfidf_for_column(df, column)
        tfidf_data = {"matrix": tfidf_matrix, "feature_names": feature_names}
        save_tfidf_data(tfidf_data, f'tfidf_{column}.pkl')
    
    print("All TF-IDF computations completed.")

if __name__ == "__main__":
    main()