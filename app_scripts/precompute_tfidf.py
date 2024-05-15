import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import os
import pickle

def load_data():
    """
    Load the dataset from the 'data' directory.
    Returns:
    DataFrame: The loaded data.
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CMFA_PressCon_v4.xlsx')
    return pd.read_excel(file_path)

def preprocess_data(data):
    """
    Process data to combine lemmatized answers into a single text corpus per document.
    Returns:
    Series: A pandas Series containing preprocessed text data.
    """
    if 'answer_lem' in data.columns:
        return data['answer_lem'].fillna('')
    else:
        raise ValueError("Column 'answer_lem' does not exist in the data.")

def compute_yearly_tfidf(data):
    """
    Compute and store comprehensive TF-IDF scores for each year.
    """
    results = {}
    for year in data['year'].unique():
        yearly_data = data[data['year'] == year]['answer_lem'].dropna().tolist()
        if not yearly_data:
            continue
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
        tfidf_matrix = vectorizer.fit_transform(yearly_data)
        feature_names = vectorizer.get_feature_names_out()
        # Store the entire TF-IDF matrix and the corresponding feature names
        results[year] = (tfidf_matrix, feature_names)
    return results

def save_yearly_tfidf(results):
    """
    Save the computed yearly TF-IDF matrices and feature names to a pickle file.
    """
    with open('yearly_tfidf_full.pkl', 'wb') as f:
        pickle.dump(results, f)

def main():
    """
    Main function to orchestrate the loading, processing, computation, and saving of TF-IDF results.
    """
    data = load_data()
    corpus = preprocess_data(data)
    yearly_tfidf_scores = compute_yearly_tfidf(data)
    save_yearly_tfidf(yearly_tfidf_scores)

if __name__ == '__main__':
    main()
