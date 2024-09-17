import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from scipy.sparse import load_npz
import pickle
import time
from collections import Counter
import calendar
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

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
    
    # Convert 'date' column to datetime, replacing invalid entries with NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with NaT values in the 'date' column
    df = df.dropna(subset=['date'])
    
    # Ensure 'year', 'month', and 'day' columns are correct
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
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
def load_tfidf_data(tfidf_label, group_by):
    """
    Load the TF-IDF data from the specified pickle file based on max_df setting and group_by (year or month).
    """
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', f'{group_by}_tfidf_maxdf{tfidf_label}.pkl')
    with open(file_path, 'rb') as file:
        _tfidf_data = pickle.load(file)
    _tfidf_data = {str(k): v for k, v in _tfidf_data.items()}
    return _tfidf_data

@st.cache_data
def load_entities_tfidf():
    base_path = os.path.dirname(__file__)
    file_paths = {
        "loc": os.path.join(base_path, '../data/tfidf_a_loc.pkl'),
        "org": os.path.join(base_path, '../data/tfidf_a_org.pkl'),
        "per": os.path.join(base_path, '../data/tfidf_a_per.pkl'),
        "misc": os.path.join(base_path, '../data/tfidf_a_misc.pkl'),
    }

    #for key, path in file_paths.items():
        #print(f"{key}: {path}")

    entities_tfidf = {}
    for key, path in file_paths.items():
        with open(path, 'rb') as file:
            entities_tfidf[key] = pickle.load(file)
            
    return entities_tfidf

def split_and_clean(text):
    if pd.isna(text):
        return []
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

def extract_relevant_tfidf(_tfidf_data, filtered_data, group_by):
    grouped_documents = filtered_data.groupby(group_by)['answer_lem'].apply(lambda x: ' '.join(x.dropna())).to_dict()
    # Convert keys to strings
    grouped_documents = {str(k): v for k, v in grouped_documents.items()}
    #print(f"Grouped documents keys: {list(grouped_documents.keys())[:5]}")
    #print(f"TF-IDF data keys: {list(_tfidf_data.keys())[:5]}")
    tfidf_scores = {str(group): {} for group in _tfidf_data.keys()}
    sentiment_scores = {str(group): {} for group in _tfidf_data.keys()}

    for group, document in grouped_documents.items():
        group_str = str(group)
        if group_str in _tfidf_data:
            matrix, feature_names, precomputed_sentiments = _tfidf_data[group_str]
            feature_names_list = list(feature_names)
            document_terms = document.split()
            term_indices = [feature_names_list.index(term) for term in document_terms if term in feature_names_list]
            if term_indices:
                relevant_matrix = matrix[:, term_indices]
                summed_scores = np.array(relevant_matrix.sum(axis=0)).flatten()
                tfidf_scores[group_str] = dict(zip([feature_names_list[i] for i in term_indices], summed_scores))
                sentiment_scores[group_str] = {term: precomputed_sentiments.get(term, np.nan) for term in document_terms}
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

    # Sort and get top 10 terms and their scores
    sorted_items = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    top_terms = sorted_items[:10]  # List of (term, score)
    
    return top_terms

def display_top_entities_tfidf(filtered_data):

    # Only proceed if there is valid filtered data
    if filtered_data is None or filtered_data.empty:
        st.write("No data available for the selected criteria.")
        return

    entities_tfidf = load_entities_tfidf()  # Load all entities TF-IDF data

    # Prepare dictionaries to store top entities and their scores
    top_entities = {}
    categories = ['loc', 'org', 'per', 'misc']
    category_names = {'loc': 'Location', 'org': 'Organization', 'per': 'Person', 'misc': 'Keyword'}
    category_colors = {'loc': 'red', 'org': 'blue', 'per': 'green', 'misc': 'yellow'}

    # For each category, get top terms and their scores
    for cat in categories:
        top_terms_scores = get_top_tfidf_terms(entities_tfidf[cat], filtered_data, f'a_{cat}')
        top_entities[cat] = top_terms_scores  # List of (term, score)

    # Build the node list
    nodes = []
    node_sizes = []
    node_colors = []
    node_texts = []
    node_categories = {}
    term_set = set()
    for cat in categories:
        for term, score in top_entities[cat]:
            term_normalized = term.lower().strip()
            nodes.append(term_normalized)
            node_sizes.append(score)
            node_colors.append(category_colors[cat])
            node_texts.append(f"{term} ({category_names[cat]})")
            node_categories[term] = cat
            term_set.add(term_normalized)
    
    # Build the graph with weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # Initialize edge weights
    for idx, row in filtered_data.iterrows():
        entities_in_row = []
        for cat in categories:
            col_name = f'a_{cat}'
            if pd.notna(row[col_name]):
                entities = split_and_clean(row[col_name])
                entities = [e.lower().strip() for e in entities]
                # Filter to include only top entities
                entities = [e for e in entities if e in term_set]
                entities_in_row.extend(entities)
        # Add edges between all pairs of entities in this row
        for i in range(len(entities_in_row)):
            for j in range(i+1, len(entities_in_row)):
                e1 = entities_in_row[i]
                e2 = entities_in_row[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]['weight'] += 1
                else:
                    G.add_edge(e1, e2, weight=1)
                    
    
    # Normalize edge weights for layout algorithms
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    max_weight = max(weights) if weights else 1
    for u, v, d in G.edges(data=True):
        d['weight'] = d['weight'] / max_weight

    # Compute the layout using 'graphviz_layout_neato'
    try:
        pos = graphviz_layout(G, prog='neato')
    except Exception as e:
        st.error(f"Error with Graphviz layout 'neato': {e}")
        st.warning("Falling back to spring layout.")
        pos = nx.spring_layout(G, weight='weight', seed=42)

    
    # Store positions in node attributes for easy access
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_counter = 0
    for u, v, d in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_counter += 1

    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Prepare node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    # Normalize node sizes for visualization
    max_size = max(node_sizes) if node_sizes else 1
    node_sizes_normalized = [ (s / max_size) * 50 + 10 for s in node_sizes ]  # Scale sizes between 10 and 60

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_texts,
        marker=dict(
            size=node_sizes_normalized,
            color=node_colors,
            line_width=2,
        )
    )

    # Create figure and add both edge and node traces
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Key Associations Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    st.plotly_chart(fig, use_container_width=True)
    
        
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
    
    # Add a toggle to select the time granularity
    time_granularity = st.radio(
        "Select Time Granularity:",
        ('Yearly', 'Monthly'),
        index=0,
        help="Choose 'Yearly' to view data aggregated by year or 'Monthly' for data aggregated by month."
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

ggroup_by = 'year' if time_granularity == 'Yearly' else 'month'
def plot_combined_timeline(filtered_data, overall_data, group_by):
    if group_by == 'month':
        filtered_data['month'] = pd.to_datetime(filtered_data['date']).dt.to_period('M')
        overall_data['month'] = pd.to_datetime(overall_data['date']).dt.to_period('M')
        
        timeline_data = filtered_data.groupby('month').size().reset_index(name="Counts")
        sentiment_by_location = filtered_data.groupby('month')["a_sentiment"].mean().reset_index()
        overall_sentiment = overall_data.groupby('month')["a_sentiment"].mean().reset_index()
        
        # Convert month periods to datetime for proper plotting
        timeline_data['month'] = timeline_data['month'].dt.to_timestamp()
        sentiment_by_location['month'] = sentiment_by_location['month'].dt.to_timestamp()
        overall_sentiment['month'] = overall_sentiment['month'].dt.to_timestamp()
        
        # Sort the dataframes by date
        timeline_data = timeline_data.sort_values('month')
        sentiment_by_location = sentiment_by_location.sort_values('month')
        overall_sentiment = overall_sentiment.sort_values('month')
        
        # Create month labels (e.g., 'Jan '21')
        month_labels = [date.strftime('%b \'%y') for date in timeline_data['month']]
        
        # Now, create tickvals and ticktext for every 6th month
        tick_indices = list(range(0, len(month_labels), 6))
        tickvals = [timeline_data['month'].iloc[i] for i in tick_indices]
        ticktext = [month_labels[i] for i in tick_indices]
        
        x_values = timeline_data['month']
    else:
        # Yearly data
        timeline_data = filtered_data.groupby('year').size().reset_index(name="Counts")
        sentiment_by_location = filtered_data.groupby('year')["a_sentiment"].mean().reset_index()
        overall_sentiment = overall_data.groupby('year')["a_sentiment"].mean().reset_index()
        
        # Ensure group labels are strings for consistency
        x_values = timeline_data['year'].astype(str)
        sentiment_by_location['year'] = sentiment_by_location['year'].astype(str)
        overall_sentiment['year'] = overall_sentiment['year'].astype(str)
        month_labels = x_values
        tickvals = x_values
        ticktext = x_values

    # Prepare the x-axis title based on group_by
    x_axis_title = "Year" if group_by == "year" else "Month"
    
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=x_values, y=timeline_data["Counts"], name="Entry Counts"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sentiment_by_location["a_sentiment"],
            name="Sentiment for Query",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=overall_sentiment["a_sentiment"],
            name="Overall Average Sentiment",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title="Counts",
        yaxis2_title="Average Sentiment",
        width=1200,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
        )
    )
    fig.update_yaxes(title_text="Entry Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=True)
    st.plotly_chart(fig, use_container_width=False)

def display_tfidf_scores(filtered_data, overall_data, group_by):
    #print("Displaying TF-IDF scores...")
    #start_time = time.time()
    
    _tfidf_data = load_tfidf_data(tfidf_label)
    
    # Compute overall sentiment
    overall_sentiment = overall_data.groupby('year')['a_sentiment'].mean().to_dict()
    
    tfidf_df, sentiment_scores = extract_relevant_tfidf(_tfidf_data, filtered_data, overall_sentiment)
    
    if not tfidf_df.empty:
        formatted_df = pd.DataFrame()

        for group in tfidf_df.columns:
            top_terms = tfidf_df[group].dropna().sort_values(ascending=False).head(10)
            if not top_terms.empty:
                max_score = top_terms.iloc[0]
                group_sentiment_scores = sentiment_scores[group]
                colors = assign_colors_dynamically(group_sentiment_scores, st.session_state.overall_sentiment[group])
                formatted_terms = []
                for term, score in top_terms.items():
                    color = colors.get(term, 'white')
                    formatted_term = f"<span style='color:{color}'>{term} ({(score/max_score * 100):.2f}%)</span>"
                    formatted_terms.append(formatted_term)
                formatted_df[group] = pd.Series(formatted_terms).reset_index(drop=True)

        formatted_df = formatted_df.dropna(how='all', axis=1)

        if not formatted_df.empty:
            st.markdown(formatted_df.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write("All years resulted in empty data after processing.")
    else:
        st.write("No relevant TF-IDF scores found for the selected query.")
        
    #print(f"TF-IDF score display completed in {time.time() - start_time} seconds")

if analyze_button:
    st.session_state.filtered_data = filter_data(selected_terms, logic_type)
    #print(f"Filtered data has {len(st.session_state.filtered_data)} rows.")

    # Proceed only if filtered_data is not empty
    if st.session_state.filtered_data.empty:
        st.error("No data available after filtering with the selected criteria.")
        st.session_state.display_qa_pairs = False
    else:
        st.session_state.display_qa_pairs = True
        max_df_value = 20  # Hard-coded max_df value (20%)
        group_by = 'year' if time_granularity == 'Yearly' else 'month'

        # Ensure 'month' is correctly formatted
        if group_by == 'month':
            data['month'] = data['date'].dt.strftime('%Y-%m')
            st.session_state.filtered_data['month'] = st.session_state.filtered_data['date'].dt.strftime('%Y-%m')
            #print(f"Data 'month' column sample: {data['month'].head()}")
            #print(f"Filtered data 'month' column sample: {st.session_state.filtered_data['month'].head()}")
        else:
            data['year'] = data['year'].astype(str)
            st.session_state.filtered_data['year'] = st.session_state.filtered_data['year'].astype(str)

        tfidf_data = load_tfidf_data(max_df_value, group_by)
        #print(f"TF-IDF data keys: {list(tfidf_data.keys())[:5]}")  # Print first 5 keys
        
        st.session_state.overall_sentiment = data.groupby(group_by)['a_sentiment'].mean().to_dict()
        # Convert keys to strings
        st.session_state.overall_sentiment = {str(k): v for k, v in st.session_state.overall_sentiment.items()}

        st.session_state.tfidf_df, st.session_state.sentiment_scores = extract_relevant_tfidf(
            _tfidf_data=tfidf_data,
            filtered_data=st.session_state.filtered_data,
            group_by=group_by
        )

group_by = 'year' if time_granularity == 'Yearly' else 'month'

# Render timeline of mentions and sentiment over time if filtered data is available
if st.session_state.filtered_data is not None:
    with st.expander("📈 View Timeline of Mentions and Sentiment Over Time", expanded=True):
        plot_combined_timeline(st.session_state.filtered_data, data, group_by)
        display_basic_stats()

# Key Associations: Discover the Top Locations, Organizations, and Individuals Linked by China’s MFA to Your Query
if st.session_state.filtered_data is not None:
    with st.expander("🕸️ Key Associations: Discover the Top Locations, Organizations, and Individuals Linked by China’s MFA to Your Query", expanded=False):
        display_top_entities_tfidf(st.session_state.filtered_data)

        st.markdown("""
            **What does this network represent?**

            - **Nodes**: Each node represents a key term or entity (e.g., locations, organizations, people, keywords) that is frequently associated with your selected query.
              - **Color**: Represents the category of the entity:
                - **Red**: Location
                - **Blue**: Organization
                - **Green**: Person
                - **Yellow**: Keyword
              - **Size**: Indicates the importance or frequency of the term in the context of your query. Larger nodes are more significant.
            - **Edges**: The lines connecting nodes represent associations between terms. An edge between two nodes means that the terms frequently appear together in the same context.
              - **Thickness**: Represents the strength of the association. Thicker edges indicate stronger associations.
            - **Layout**: The placement of nodes is determined by the 'neato' algorithm, which positions closely related nodes nearer to each other. This helps to visualize clusters of related terms.

            This network helps you understand how different key terms are interconnected in the Chinese MFA's statements regarding your query.
        """)

# Uncover key terms in China's MFA statements
if st.session_state.tfidf_df is not None:
    with st.expander("🔍 Uncover Key Terms in China's MFA Statements (only for yearly analysis)", expanded=False):
        tfidf_df = st.session_state.tfidf_df
        sentiment_scores = st.session_state.sentiment_scores

        # Display the results
        if not tfidf_df.empty:
            formatted_df = pd.DataFrame()

            for group in tfidf_df.columns:
                group = str(group)  # Ensure group is a string
                top_terms = tfidf_df[group].dropna().sort_values(ascending=False).head(10)
                if not top_terms.empty:
                    max_score = top_terms.iloc[0]
                    group_sentiment_scores = sentiment_scores[group]

                    # Dynamically assign colors based on sentiment distribution
                    colors = assign_colors_dynamically(group_sentiment_scores, st.session_state.overall_sentiment[group])

                    formatted_terms = []
                    for term, score in top_terms.items():
                        color = colors.get(term, 'white')
                        formatted_term = f"<span style='color:{color}'>{term} ({(score/max_score * 100):.2f}%)</span>"
                        formatted_terms.append(formatted_term)

                    formatted_df[group] = pd.Series(formatted_terms).reset_index(drop=True)

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

data['date'] = pd.to_datetime(data['date'], errors='coerce')

if st.session_state.display_qa_pairs:
    if time_granularity == 'Yearly':
        min_year = int(st.session_state.filtered_data["year"].min())
        max_year = int(st.session_state.filtered_data["year"].max())
        selected_range = st.slider(
            "Select Year Range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="year_slider"
        )
        # Filter data based on selected range
        group_data = st.session_state.filtered_data[
            (st.session_state.filtered_data["year"].astype(int) >= selected_range[0]) &
            (st.session_state.filtered_data["year"].astype(int) <= selected_range[1])
        ]
    else:
        st.session_state.filtered_data['month'] = st.session_state.filtered_data['date'].dt.to_period('M')
        # Convert Period to Timestamp, then to Python datetime
        min_month = st.session_state.filtered_data["month"].min().to_timestamp().to_pydatetime()
        max_month = st.session_state.filtered_data["month"].max().to_timestamp().to_pydatetime()
        selected_range = st.slider(
            "Select Month Range:",
            min_value=min_month,
            max_value=max_month,
            value=(min_month, max_month),
            format="MMM YYYY",
            key="month_slider"
        )
        # Filter data based on selected range
        group_data = st.session_state.filtered_data[
            (st.session_state.filtered_data["date"] >= selected_range[0]) &
            (st.session_state.filtered_data["date"] <= selected_range[1])
        ]

    if not group_data.empty:
        display_qa_pairs(group_data)
    else:
        st.error("No question-answer pairs to display for the selected period.")