import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from scipy.sparse import load_npz
import pickle
import time
from collections import Counter

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="China MFA Dashboard",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwHrb5on9YN6Abs7KvAuvz5M4dxTJLDfMT7w&s",
)

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', 'CMFA_PressCon_v4.xlsx')
    
    columns_to_load = [
        "id", "day", "month", "year", "date", "question", "answer",
        "question_lem", "answer_lem", "q_loc", "q_per", "q_org", 
        "q_misc", "a_loc", "a_per", "a_org", "a_misc", 
        "a_sentiment", "q_sentiment"
    ]
    
    # loading only necessary columns
    df = pd.read_excel(file_path, usecols=columns_to_load)
    
    columns_to_clean = ["a_per", "a_loc", "a_org", "a_misc"]
    for column in columns_to_clean:
        df[column] = df[column].replace("-", np.nan).astype(str)
    
    return df

data = load_data()

@st.cache_data
def load_precomputed_stats():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', 'precomputed_stats.pkl')
    stats_df = pd.read_pickle(file_path)
    return stats_df

precomputed_stats = load_precomputed_stats()

# Functions to load TF-IDF components
@st.cache_data
def load_tfidf_data(tfidf_label):
    """
    Load the TF-IDF data from the specified pickle file based on user-selected max_df setting.
    """
    #print(f"Loading TF-IDF data for max_df={tfidf_label}...")
    #start_time = time.time()
    
    base_path = os.path.dirname(__file__)
    # adjusting filename based on slider input by user (corresponds to the file suffix)
    file_path = os.path.join(base_path, '..', 'data', f'yearly_tfidf_maxdf{tfidf_label}.pkl')
    with open(file_path, 'rb') as file:
        _tfidf_data = pickle.load(file)
    #print(f"TF-IDF data loaded in {time.time() - start_time} seconds")
    return _tfidf_data

@st.cache_data
def load_entities_tfidf():
    base_path = os.path.dirname(__file__)
    print(f"Attempting to load the following TF-IDF files:")
    file_paths = {
        "loc": os.path.join(base_path, '../data/tfidf_a_loc.pkl'),
        "org": os.path.join(base_path, '../data/tfidf_a_org.pkl'),
        "per": os.path.join(base_path, '../data/tfidf_a_per.pkl'),
        "misc": os.path.join(base_path, '../data/tfidf_a_misc.pkl'),
    }

    for key, path in file_paths.items():
        print(f"{key}: {path}")

    entities_tfidf = {}
    for key, path in file_paths.items():
        with open(path, 'rb') as file:
            entities_tfidf[key] = pickle.load(file)
            print(f"Loaded {key}: {len(entities_tfidf[key])} items")
   
    print("TF-IDF data for locations:", entities_tfidf['loc'])
    print("TF-IDF data for organizations:", entities_tfidf['org'])
    print("TF-IDF data for people:", entities_tfidf['per'])
    print("TF-IDF data for miscellaneous:", entities_tfidf['misc'])
    
    return entities_tfidf

def split_and_clean(text):
    return [
        t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"
    ]

def convert_sparse_matrix_to_dict(tfidf_data):
    """Convert sparse matrix to a dictionary with terms as keys and tfidf scores as values."""
    matrix = tfidf_data['matrix']
    feature_names = tfidf_data['feature_names']
    
    term_tfidf_dict = {}
    
    # Iterate over each feature (term)
    for i, term in enumerate(feature_names):
        # Extract the column (i.e., tfidf scores for that term across all documents)
        col = matrix[:, i].toarray().flatten()
        # Sum tfidf scores for this term across all documents
        term_tfidf_dict[term] = col.sum()
    
    return term_tfidf_dict

# Compute term frequencies across all categories
term_frequencies = Counter()
term_original_forms = {}

for column in ['a_per', 'a_org', 'a_loc', 'a_misc']:
    terms = data[column].dropna().apply(split_and_clean).tolist()
    for sublist in terms:
        for term in sublist:
            term_lower = term.lower()
            term_frequencies[term_lower] += 1
            term_original_forms[term_lower] = term  # Keep original capitalization

# Create a combined list of all search values sorted by frequency
all_search_values = [term_original_forms[term] for term, freq in term_frequencies.most_common()]

@st.cache_data
def filter_data(selected_terms, logic_type):
    if not selected_terms:
        return data

    def term_in_any_column(row, term):
        return any(term in split_and_clean(row[col]) if pd.notna(row[col]) else False for col in ['a_per', 'a_org', 'a_loc', 'a_misc'])

    if logic_type == 'AND':
        return data[data.apply(lambda row: all(term_in_any_column(row, term) for term in selected_terms), axis=1)]
    else:
        return data[data.apply(lambda row: any(term_in_any_column(row, term) for term in selected_terms), axis=1)]

if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'display_qa_pairs' not in st.session_state:
    st.session_state.display_qa_pairs = False
if 'tfidf_df' not in st.session_state:
    st.session_state.tfidf_df = None
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = None
if 'overall_sentiment' not in st.session_state:
    st.session_state.overall_sentiment = None  # Initialize overall_sentiment

def assign_colors_dynamically(sentiment_scores, overall_sentiment):
    #sentiment scores to a list and removing any NaN values
    sentiments = np.array(list(sentiment_scores.values()))
    sentiments = sentiments[~np.isnan(sentiments)]

    #25th, 50th (median), and 75th percentiles
    q25, q50, q75 = np.percentile(sentiments, [25, 50, 75])

    #colors based on where the term's sentiment score falls
    colors = {}
    for term, score in sentiment_scores.items():
        if score < q25:
            colors[term] = 'red'
        elif score > q75:
            colors[term] = 'green'
        else:
            colors[term] = 'grey'
    
    return colors

def extract_relevant_tfidf(_tfidf_data, filtered_data):
    #print("Extracting relevant TF-IDF scores...")
    #start_time = time.time()

    yearly_documents = filtered_data.groupby('year')['answer_lem'].apply(lambda x: ' '.join(x.dropna())).to_dict()
    tfidf_scores = {year: {} for year in _tfidf_data.keys()}
    sentiment_scores = {year: {} for year in _tfidf_data.keys()}

    for year, document in yearly_documents.items():
        if year in _tfidf_data:
            matrix, feature_names, precomputed_sentiments = _tfidf_data[year]
            feature_names_list = list(feature_names)
            document_terms = document.split()
            term_indices = [feature_names_list.index(term) for term in document_terms if term in feature_names_list]
            if term_indices:
                relevant_matrix = matrix[:, term_indices]
                summed_scores = np.array(relevant_matrix.sum(axis=0)).flatten()
                tfidf_scores[year] = dict(zip([feature_names_list[i] for i in term_indices], summed_scores))
                
                #retrieving precomputed average sentiment for each term
                sentiment_scores[year] = {term: precomputed_sentiments.get(term, np.nan) for term in document_terms}

    #print(f"TF-IDF extraction completed in {time.time() - start_time} seconds")
    return pd.DataFrame(tfidf_scores), sentiment_scores

def normalize_term(term):
    """Normalize a term by lowercasing and stripping whitespace."""
    if isinstance(term, str):
        return term.lower().strip()
    return term

def get_top_tfidf_terms(tfidf_data, filtered_data, column_name):
    # Normalize and extract terms in the filtered data
    terms_in_filtered_data = filtered_data[column_name].dropna().apply(split_and_clean).explode().tolist()
    terms_in_filtered_data = [normalize_term(term) for term in terms_in_filtered_data]

    tfidf_scores = {}
    feature_names = [normalize_term(term) for term in tfidf_data['feature_names']]

    for term in terms_in_filtered_data:
        if term in feature_names:
            index = feature_names.index(term)
            # Sum the TF-IDF scores across all documents that mention the term
            tfidf_scores[term] = tfidf_data['matrix'][:, index][filtered_data.index].sum()

    sorted_items = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    top_terms = [term for term, score in sorted_items[:10]]
    
    return top_terms

def display_top_entities_tfidf(filtered_data):
    col1, col2, col3, col4 = st.columns(4)

    # Only proceed if there is valid filtered data
    if filtered_data is None or filtered_data.empty:
        st.write("No data available for the selected criteria.")
        return

    entities_tfidf = load_entities_tfidf()  # Load all entities TF-IDF data

    all_criteria = []
    combined_criteria = ", ".join(selected_terms) if selected_terms else "selected criteria"
    category_title = f"associated with '{combined_criteria}'"

    with col1:
        top_locations = get_top_tfidf_terms(entities_tfidf['loc'], filtered_data, 'a_loc')
        locations_df = pd.DataFrame(top_locations, columns=["Location"])
        st.table(locations_df.style.hide(axis="index"))

    with col2:
        top_organizations = get_top_tfidf_terms(entities_tfidf['org'], filtered_data, 'a_org')
        organizations_df = pd.DataFrame(top_organizations, columns=["Organization"])
        st.table(organizations_df.style.hide(axis="index"))

    with col3:
        top_people = get_top_tfidf_terms(entities_tfidf['per'], filtered_data, 'a_per')
        people_df = pd.DataFrame(top_people, columns=["Person"])
        st.table(people_df.style.hide(axis="index"))

    with col4:
        top_misc = get_top_tfidf_terms(entities_tfidf['misc'], filtered_data, 'a_misc')
        misc_df = pd.DataFrame(top_misc, columns=["Keyword"])
        st.table(misc_df.style.hide(axis="index"))
        
def display_basic_stats():
    messages = []
    for term in selected_terms:
        processed_term = term.lower()
        found = False
        for category in ['a_per', 'a_loc', 'a_org', 'a_misc']:
            category_stats = precomputed_stats.get(category, {})
            term_stats = category_stats.get(processed_term)
            if term_stats:
                rank = term_stats['rank']
                category_name = {
                    "a_per": "People mentioned by the Chinese MFA",
                    "a_loc": "Locations mentioned by the Chinese MFA",
                    "a_org": "Organizations mentioned by the Chinese MFA",
                    "a_misc": "Miscellaneous terms mentioned by the Chinese MFA"
                }[category]
                message = f"🌟 {term.capitalize()} is the #{rank} most common value in {category_name}. 🌟"
                messages.append(message)
                found = True
                break
        if not found:
            pass  # Term not found in any category

    # Display all the messages below the timeline plot
    if messages:
        st.markdown("#### Key Statistics")
        for msg in messages:
            st.markdown(msg)

def plot_frequency_over_time(term, category):
    yearly_data = data[data[category].str.contains(term, regex=False, na=False)]
    yearly_counts = yearly_data.groupby(yearly_data['year']).size()
    total_counts = data.groupby('year').size()
    
    yearly_frequencies = (yearly_counts / total_counts) * 100
    
    fig = px.bar(yearly_frequencies, labels={'value': '% of Entries', 'year': 'Year'},
                 title=f'Frequency of "{term}" Over Time in {category}')
    st.plotly_chart(fig, use_container_width=True)

def display_qa_pairs(year_data):
    st.write("Question and Answer Pairs with Sentiment Score")
    columns_to_display = ['date', 'question', 'q_sentiment', 'answer', 'a_sentiment', 'a_loc', 'a_per', 'a_org', 'a_misc']
    column_rename_mapping = {
        'q_sentiment': 'Question Sentiment',
        'a_sentiment': 'Answer Sentiment',
        'a_loc': 'Answer Locations',
        'a_per': 'Answer Persons',
        'a_org': 'Answer Organizations',
        'a_misc': 'Answer Other Keywords',
    }
    display_df = year_data[columns_to_display].rename(columns=column_rename_mapping)
    st.dataframe(display_df, key="qa_pairs")
    
with st.sidebar:
    st.image("https://www.aies.at/img/layout/AIES-Logo-EN-white.png?m=1684934843", use_column_width=True)
    st.title("What does the official China say about...?")
    st.markdown(
        """
    This AIES interactive dashboard lets you explore a corpus of the press conferences by the Chinese Ministry of Foreign Affairs. The dataset is a unique source of information covering 20+ years of China's foreign policy discourse. Select different criteria to get insights from the data.
    
    A big thank you to Richard Turcsányi for his input! Data source: https://doi.org/10.1007/s11366-021-09762-3
    """
    )

    st.title("Select Search Criteria")
    selected_terms = st.multiselect("Select Search Criteria:", all_search_values, key='select_terms')
    logic_type = st.radio(
        "Select Logic Type:",
        ('AND', 'OR'),
        index=1,
        help="Choose 'AND' to display entries that meet all criteria or 'OR' for entries that meet any of the selected criteria."
    )
    
    # CSS for button styling
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #023059; /* Dark Blue */
            color: white;
            font-weight: bold;
            font-size: 18px;
            height: 3em;
            width: 100%;
            border-radius: 10px;
            border: 2px solid #023059;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
        }
        div.stButton > button:first-child:hover {
            background-color: #F29A2E; /* Orange */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    analyze_button = st.button("Analyze")
    
    st.markdown(
        """
        <div style="height: 30px;"></div>

        How to cite: *Urosevic, A. (2024), Interactive China MFA Dashboard, Austrian Institute for European and Security Policy. Available at: https://www.aies.at/china-dashboard*
        """,
        unsafe_allow_html=True
    )

filtered_data = filter_data(selected_terms, logic_type)

def plot_combined_timeline(filtered_data, overall_data):
    #print("Plotting combined timeline...")
    #start_time = time.time()
    
    timeline_data = filtered_data.groupby("year").size().reset_index(name="Counts")

    sentiment_by_location = (
        filtered_data.groupby("year")["a_sentiment"].mean().reset_index()
    )
    overall_sentiment = overall_data.groupby("year")["a_sentiment"].mean().reset_index()

    from plotly.subplots import make_subplots
    import plotly.graph_objs as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=timeline_data["year"], y=timeline_data["Counts"], name="Entry Counts"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=sentiment_by_location["year"],
            y=sentiment_by_location["a_sentiment"],
            name="Sentiment for Query",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=overall_sentiment["year"],
            y=overall_sentiment["a_sentiment"],
            name="Overall Average Sentiment",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Counts",
        yaxis2_title="Average Sentiment",
        width=1200,
        height=600,
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Anchor the legend at the top
            y=-0.2,  # Position it slightly below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.5  # Position it at the center of the plot
        )
    )

    fig.update_yaxes(title_text="Entry Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=True)

    st.plotly_chart(fig, use_container_width=False)
    
    #print(f"Timeline plotting completed in {time.time() - start_time} seconds")

def display_tfidf_scores(filtered_data, overall_data):
    #print("Displaying TF-IDF scores...")
    #start_time = time.time()
    
    _tfidf_data = load_tfidf_data(tfidf_label)
    
    # Compute overall sentiment
    overall_sentiment = overall_data.groupby('year')['a_sentiment'].mean().to_dict()
    
    tfidf_df, sentiment_scores = extract_relevant_tfidf(_tfidf_data, filtered_data, overall_sentiment)
    
    if not tfidf_df.empty:
        formatted_df = pd.DataFrame()

        for year in tfidf_df.columns:
            top_terms = tfidf_df[year].dropna().sort_values(ascending=False).head(10)
            if not top_terms.empty:
                max_score = top_terms.iloc[0]
                year_sentiment_scores = sentiment_scores[year]

                # Dynamically assign colors
                colors = assign_colors_dynamically(year_sentiment_scores, overall_sentiment[year])

                formatted_terms = []
                for term, score in top_terms.items():
                    color = colors.get(term, 'white')  # Default to white if not found
                    formatted_term = f"<span style='color:{color}'>{term} ({(score/max_score * 100):.2f}%)</span>"
                    formatted_terms.append(formatted_term)

                formatted_df[year] = pd.Series(formatted_terms).reset_index(drop=True)

        formatted_df = formatted_df.dropna(how='all', axis=1)

        if not formatted_df.empty:
            st.markdown(formatted_df.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write("All years resulted in empty data after processing.")
    else:
        st.write("No relevant TF-IDF scores found for the selected query.")
        
    #print(f"TF-IDF score display completed in {time.time() - start_time} seconds")

if analyze_button:
    # Filter the data based on user selections and store in session state
    st.session_state.filtered_data = filter_data(
        selected_terms, 
        logic_type
    )
    st.session_state.display_qa_pairs = True  # Trigger the QA pairs display logic

    # Load TF-IDF data
    max_df_value = 20  # Hard-coded max_df value (20%)
    tfidf_data = load_tfidf_data(max_df_value)

    # Compute overall sentiment and extract relevant TF-IDF scores
    st.session_state.overall_sentiment = data.groupby('year')['a_sentiment'].mean().to_dict()  # Store in session state
    st.session_state.tfidf_df, st.session_state.sentiment_scores = extract_relevant_tfidf(
        _tfidf_data=tfidf_data, 
        filtered_data=st.session_state.filtered_data
    )


# Render timeline of mentions and sentiment over time if filtered data is available
if st.session_state.filtered_data is not None:
    with st.expander("📈 View Timeline of Mentions and Sentiment Over Time", expanded=True):
        plot_combined_timeline(st.session_state.filtered_data, data)
        display_basic_stats()

# Key Associations: Discover the Top Locations, Organizations, and Individuals Linked by China’s MFA to Your Query
if st.session_state.filtered_data is not None:
    with st.expander("🕸️ Key Associations: Discover the Top Locations, Organizations, and Individuals Linked by China’s MFA to Your Query", expanded=False):
        display_top_entities_tfidf(st.session_state.filtered_data)

# Uncover key terms in China's MFA statements
if st.session_state.tfidf_df is not None:
    with st.expander("🔍 Uncover Key Terms in China's MFA Statements", expanded=False):
        tfidf_df = st.session_state.tfidf_df
        sentiment_scores = st.session_state.sentiment_scores

        # Display the results
        if not tfidf_df.empty:
            formatted_df = pd.DataFrame()

            for year in tfidf_df.columns:
                top_terms = tfidf_df[year].dropna().sort_values(ascending=False).head(10)
                if not top_terms.empty:
                    max_score = top_terms.iloc[0]
                    year_sentiment_scores = sentiment_scores[year]

                    # Dynamically assign colors based on sentiment distribution
                    colors = assign_colors_dynamically(year_sentiment_scores, st.session_state.overall_sentiment[year])

                    formatted_terms = []
                    for term, score in top_terms.items():
                        color = colors.get(term, 'white')  # Default to white if not found
                        formatted_term = f"<span style='color:{color}'>{term} ({(score/max_score * 100):.2f}%)</span>"
                        formatted_terms.append(formatted_term)

                    formatted_df[year] = pd.Series(formatted_terms).reset_index(drop=True)

            formatted_df = formatted_df.dropna(how='all', axis=1)

            if not formatted_df.empty:
                st.markdown(formatted_df.to_html(escape=False), unsafe_allow_html=True)

                # Display the help text below the DataFrame
                st.markdown(
                    """
                    #### Explanation of the Key Term Results above:
                    
                    This section identifies the most significant terms in the Chinese Ministry of Foreign Affairs' 
                    responses for the selected criteria. These terms are not just frequent; they are unusually common 
                    in the selected responses compared to all other responses from China in that year. This ranking is 
                    done using a technique called TF-IDF (Term Frequency-Inverse Document Frequency), which highlights 
                    terms that are especially relevant in the context of your query.

                    **Note:** Only terms that appear in 20% or fewer of all responses provided by the MFA are considered. 
                    This filter helps to avoid overly generic terms, such as "China," that are less useful for identifying 
                    key themes specific to your query.

                    **Color Coding**:
                    
                    - **Green**: Sentiment is in the top 25th percentile (more positive than the overall average).
                    - **Red**: Sentiment is in the bottom 25th percentile (less positive than the overall average).
                    - **Neutral (White)**: Sentiment falls within the middle 50th percentile, close to the overall average.
                    """
                )
            else:
                st.write("No relevant terms found for the selected query.")
        else:
            st.write("No relevant terms found for the selected query.")
            
            
# CSS for adding spacing between elements
st.markdown(
    """
    <style>
    .spacer {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a spacer between buttons and the next section
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

if st.session_state.display_qa_pairs:
    # Add an arrow and text to guide users to scroll down
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <span style='font-size: 24px;'>⬇️ Dive into the full Questions and Answers ⬇️</span>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.display_qa_pairs:
    selected_year = st.select_slider(
        "Select a Year:", 
        options=sorted(st.session_state.filtered_data["year"].unique()), 
        key="year_slider"
    )

    year_data = st.session_state.filtered_data[st.session_state.filtered_data["year"] == selected_year]

    if not year_data.empty:
        display_qa_pairs(year_data)
    else:
        st.error("No question-answer pairs to display for the selected year.")