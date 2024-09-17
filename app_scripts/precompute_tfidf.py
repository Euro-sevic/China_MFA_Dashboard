import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

def load_data():
    """
    Load the dataset from the 'data' directory and parse the date column.
    Returns:
    DataFrame: The loaded data with properly formatted date column.
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CMFA_PressCon_v4.xlsx')
    df = pd.read_excel(file_path, parse_dates=['date'])
    
    # If date parsing fails, try manual conversion
    if df['date'].dtype == object:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    
    print("Columns loaded:", df.columns)
    print("Date column dtype:", df['date'].dtype)
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

def compute_tfidf_with_sentiment(data, max_df, group_by):
    """
    Compute TF-IDF scores and average sentiment scores for each term per group (year or month).
    """
    results = {}
    
    # Add 'group' column based on group_by parameter
    if group_by == 'year':
        data['group'] = data['year']
    elif group_by == 'month':
        data['group'] = data['date'].dt.to_period('M')
    else:
        raise ValueError("group_by must be 'year' or 'month'")
    
    groups = data['group'].unique()
    
    for group in groups:
        group_data = data[data['group'] == group]
        lem_texts = group_data['answer_lem'].dropna().tolist()
        
        if not lem_texts:
            continue
        
        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_df=max_df)
        tfidf_matrix = vectorizer.fit_transform(lem_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Compute average sentiment for each term
        sentiment_scores = {}
        for term in feature_names:
            term_data = group_data[group_data['answer_lem'].str.contains(term, regex=False, na=False)]
            sentiment_scores[term] = term_data['a_sentiment'].mean()

        results[str(group)] = (tfidf_matrix, feature_names, sentiment_scores)
    
    return results

def save_tfidf_with_sentiment(results, max_df, group_by):
    filename = f'{group_by}_tfidf_maxdf{int(max_df*100)}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def main():
    data = load_data()
    data = preprocess_data(data)
    if 'year' not in data.columns:
        raise ValueError("Year column missing after preprocessing.")
    
    max_df_value = 0.2  # 20% threshold
    for group_by in ['year', 'month']:
        tfidf_scores = compute_tfidf_with_sentiment(data, max_df_value, group_by)
        save_tfidf_with_sentiment(tfidf_scores, max_df_value, group_by)

    # Commented out other thresholds:
    # for max_df_value in [0.5, 0.35, 0.2, 0.1, 0.05]:
    #     for group_by in ['year', 'month']:
    #         tfidf_scores = compute_tfidf_with_sentiment(data, max_df_value, group_by)
    #         save_tfidf_with_sentiment(tfidf_scores, max_df_value, group_by)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()