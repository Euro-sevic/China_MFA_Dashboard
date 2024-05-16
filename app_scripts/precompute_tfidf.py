import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

def load_data():
    """
    Load the dataset from the 'data' directory.
    Returns:
    DataFrame: The loaded data.
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CMFA_PressCon_v4.xlsx')
    df = pd.read_excel(file_path)
    print("Columns loaded:", df.columns)  # Print columns to verify
    return df


def preprocess_data(data):
    """
    Process data to combine lemmatized answers into a single text corpus per document.
    Returns:
    Series: A pandas Series containing preprocessed text data.
    """
    if 'answer_lem' in data.columns:
        return data  # Ensure the entire DataFrame is returned, not just 'answer_lem'
    else:
        raise ValueError("Column 'answer_lem' does not exist in the data.")


def compute_yearly_tfidf(data, max_df):
    """
    Compute and store comprehensive TF-IDF scores for each year with a given max_df.
    """
    results = {}
    for year in data['year'].unique():
        yearly_data = data[data['year'] == year]['answer_lem'].dropna().tolist()
        if not yearly_data:
            continue
        vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)
        tfidf_matrix = vectorizer.fit_transform(yearly_data)
        feature_names = vectorizer.get_feature_names_out()
        results[year] = (tfidf_matrix, feature_names)
    return results

def save_yearly_tfidf(results, max_df):
    """
    Save the computed yearly TF-IDF matrices and feature names to a pickle file with max_df in filename.
    """
    filename = f'yearly_tfidf_maxdf{int(max_df*100)}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def main():
    """
    Main function to orchestrate the loading, processing, computation, and saving of TF-IDF results.
    """
    data = load_data()
    data = preprocess_data(data)  # Make sure this does not convert DataFrame to Series unintentionally

    if 'year' not in data.columns:
        raise ValueError("Year column missing after preprocessing. Check preprocessing steps.")

    for max_df_value in [0.5, 0.35, 0.2]:
        yearly_tfidf_scores = compute_yearly_tfidf(data, max_df_value)
        save_yearly_tfidf(yearly_tfidf_scores, max_df_value)


if __name__ == '__main__':
    main()
