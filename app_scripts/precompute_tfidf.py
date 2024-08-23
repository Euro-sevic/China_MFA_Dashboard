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
    DataFrame: The preprocessed data with the entire DataFrame returned.
    """
    if 'answer_lem' in data.columns:
        return data
    else:
        raise ValueError("Column 'answer_lem' does not exist in the data.")

def compute_yearly_tfidf_with_sentiment(data, max_df):
    """
    Compute TF-IDF scores and average sentiment scores for each term per year.
    Returns:
    dict: A dictionary with years as keys and tuples of (TF-IDF matrix, feature names, sentiment scores) as values.
    """
    results = {}
    
    for year in data['year'].unique():
        yearly_data = data[data['year'] == year]
        lem_texts = yearly_data['answer_lem'].dropna().tolist()
        
        if not lem_texts:
            continue
        
        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)
        tfidf_matrix = vectorizer.fit_transform(lem_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Compute average sentiment for each term
        sentiment_scores = {}
        for term in feature_names:
            term_data = yearly_data[yearly_data['answer_lem'].str.contains(term, regex=False, na=False)]
            sentiment_scores[term] = term_data['a_sentiment'].mean()

        results[year] = (tfidf_matrix, feature_names, sentiment_scores)
    
    return results

def save_yearly_tfidf_with_sentiment(results, max_df):
    """
    Save the computed yearly TF-IDF matrices, feature names, and sentiment scores to a pickle file.
    """
    filename = f'yearly_tfidf_maxdf{int(max_df*100)}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def main():
    """
    Main function to orchestrate the loading, processing, computation, and saving of TF-IDF and sentiment scores.
    """
    data = load_data()
    data = preprocess_data(data)  # Make sure this does not convert DataFrame to Series unintentionally

    if 'year' not in data.columns:
        raise ValueError("Year column missing after preprocessing. Check preprocessing steps.")

    for max_df_value in [0.5, 0.35, 0.2, 0.1, 0.05]:
        yearly_tfidf_scores = compute_yearly_tfidf_with_sentiment(data, max_df_value)
        save_yearly_tfidf_with_sentiment(yearly_tfidf_scores, max_df_value)

if __name__ == '__main__':
    main()