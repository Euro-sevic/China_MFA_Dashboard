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
import io
import zipfile
import networkx.algorithms.community as nx_comm
import community.community_louvain as community_louvain
from streamlit_extras.bottom_container import bottom

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="China MFA Dashboard",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwHrb5on9YN6Abs7KvAuvz5M4dxTJLDfMT7w&s",
)

group_data = pd.DataFrame()

# I'm initializing 'time_granularity' to 'Yearly' if it's not set yet
if 'time_granularity' not in st.session_state:
    st.session_state['time_granularity'] = 'Yearly'  # Default value

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)), url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Zhongnanhai-west-wall-3436.jpg/1600px-Zhongnanhai-west-wall-3436.jpg?20081028131702");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Target the bottom container using its data-testid attribute */
    div[data-testid="stBottomBlockContainer"] {
        background: linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)), url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Zhongnanhai-west-wall-3436.jpg/1600px-Zhongnanhai-west-wall-3436.jpg?20081028131702");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding-top: 0.1px !important; /* Adjust padding */
        padding-bottom: 0.1px !important; /* Adjust padding */
        width: 101% !important; /* Make the bottom container more narrow */
        margin: 0 auto; /* Center it */
        border-radius: 1px; /* Optional: Add rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
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
    
    # Converting 'date' column to datetime, replacing invalid entries with NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Dropping rows with NaT values in the 'date' column
    df = df.dropna(subset=['date'])
    
    # Ensuring 'year', 'month', and 'day' columns are correct
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    columns_to_clean = ["a_per", "a_loc", "a_org", "a_misc"]
    for column in columns_to_clean:
        df.loc[:, column] = df[column].replace("-", np.nan).astype(str)
        
    return df

data = load_data()

@st.cache_data
def load_precomputed_stats():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', 'precomputed_stats.pkl')
    stats_df = pd.read_pickle(file_path)
    return stats_df

precomputed_stats = load_precomputed_stats()

@st.cache_data
def load_tfidf_data(tfidf_label, group_by):
    """
    Loading the TF-IDF data based on max_df setting and group_by (year or month).

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

    entities_tfidf = {}
    for key, path in file_paths.items():
        with open(path, 'rb') as file:
            entities_tfidf[key] = pickle.load(file)
            
    return entities_tfidf

st.markdown(
    """
    <style>
    /* Toggle Switch Styling */
    .toggle-switch {
        display: flex;
        align-items: center;
    }

    .toggle-switch input[type="checkbox"] {
        height: 0;
        width: 0;
        visibility: hidden;
    }

    .toggle-switch label {
        cursor: pointer;
        text-indent: -9999px;
        width: 50px;
        height: 25px;
        background: grey;
        display: block;
        border-radius: 100px;
        position: relative;
    }

    .toggle-switch label:after {
        content: '';
        position: absolute;
        top: 2px;
        left: 2px;
        width: 21px;
        height: 21px;
        background: #fff;
        border-radius: 90px;
        transition: 0.3s;
    }

    .toggle-switch input:checked + label {
        background: #F29A2E;
    }

    .toggle-switch input:checked + label:after {
        left: calc(100% - 2px);
        transform: translateX(-100%);
    }
    
    .custom-download-button > button {
        background-color: #023059 !important; /* Custom Dark Blue */
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        height: 3em !important;
        width: 100% !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2) !important;
        cursor: pointer;
    }
    .custom-download-button > button:hover {
        background-color: #F29A2E !important; /* Orange on Hover */
    }

    label:active:after {
        width: 28px;
    }
    
    table.dataframe {
        background-color: #023059 !important;
        color: white !important;
    }
    table.dataframe th {
        background-color: #023059 !important;
        color: white !important;
    }
    table.dataframe td {
        background-color: #023059 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initializing 'logic_type_toggle' if not set yet
if 'logic_type_toggle' not in st.session_state:
    st.session_state['logic_type_toggle'] = False  # Default to 'OR'

# Initializing 'monthly_granularity_toggle' if not set yet
if 'monthly_granularity_toggle' not in st.session_state:
    st.session_state['monthly_granularity_toggle'] = False  # Default to 'Yearly'

def split_and_clean(text):
    if pd.isna(text):
        return []
    return [
        t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"
    ]

def convert_sparse_matrix_to_dict(tfidf_data):
    """Converting sparse matrix to a dictionary with terms as keys and tfidf scores as values."""
    matrix = tfidf_data['matrix']
    feature_names = tfidf_data['feature_names']
    
    term_tfidf_dict = {}
    
    # Iterating over each feature (term)
    for i, term in enumerate(feature_names):
        # Extracting the column (i.e., tfidf scores for that term across all documents)
        col = matrix[:, i].toarray().flatten()
        # Summing tfidf scores for this term across all documents
        term_tfidf_dict[term] = col.sum()
    
    return term_tfidf_dict

# Computing term frequencies across all categories
term_frequencies = Counter()
term_original_forms = {}

for column in ['a_per', 'a_org', 'a_loc', 'a_misc']:
    terms = data[column].dropna().apply(split_and_clean).tolist()
    for sublist in terms:
        for term in sublist:
            term_lower = term.lower()
            term_frequencies[term_lower] += 1
            term_original_forms[term_lower] = term  # Keep original capitalization

# Creating a combined list of all search values sorted by frequency
all_search_values = [term_original_forms[term] for term, freq in term_frequencies.most_common()]

@st.cache_data
def filter_data(selected_terms, logic_type):
    if not selected_terms:
        return data

    def term_in_any_column(row, term):
        return any(term in split_and_clean(row[col]) if pd.notna(row[col]) else False for col in ['a_per', 'a_org', 'a_loc', 'a_misc'])

    if logic_type == 'AND':
        return data[data.apply(lambda row: all(term_in_any_column(row, term) for term in selected_terms), axis=1)].copy()
    else:
        return data[data.apply(lambda row: any(term_in_any_column(row, term) for term in selected_terms), axis=1)].copy()


if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = None
if 'display_qa_pairs' not in st.session_state:
    st.session_state.display_qa_pairs = False
if 'tfidf_df' not in st.session_state:
    st.session_state.tfidf_df = None
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = None
if 'overall_sentiment' not in st.session_state:
    st.session_state.overall_sentiment = None

MAX_REPORT_ENTRIES = 5
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = []

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
    # Converting keys to strings
    grouped_documents = {str(k): v for k, v in grouped_documents.items()}
    # Initializing dictionaries
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
    """Normalizing a term by lowercasing and stripping whitespace."""
    if isinstance(term, str):
        return term.lower().strip()
    return term

def get_top_tfidf_terms(tfidf_data, filtered_data, column_name):
    # Normalizing and extracting terms in the filtered data
    terms_in_filtered_data = filtered_data[column_name].dropna().apply(split_and_clean).explode().tolist()
    terms_in_filtered_data = [normalize_term(term) for term in terms_in_filtered_data]

    tfidf_scores = {}
    feature_names = [normalize_term(term) for term in tfidf_data['feature_names']]

    for term in terms_in_filtered_data:
        if term in feature_names:
            index = feature_names.index(term)
            # Summing the TF-IDF scores across all documents that mention the term
            tfidf_scores[term] = tfidf_data['matrix'][:, index][filtered_data.index].sum()

    # Sorting and getting top 10 terms and their scores
    sorted_items = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    top_terms = sorted_items[:10]  # List of (term, score)
    
    return top_terms

def display_top_entities_tfidf(filtered_data, time_period_label):
    # Only proceeding here if there is valid filtered data
    if filtered_data is None or filtered_data.empty:
        st.write(f"No data available for the selected criteria in {time_period_label}.")
        return

    entities_tfidf = load_entities_tfidf()  # Load all entities TF-IDF data

    # Preparing dictionaries to store top entities and their scores
    top_entities = {}
    categories = ['loc', 'org', 'per', 'misc']
    category_names = {'loc': 'Location', 'org': 'Organization', 'per': 'Person', 'misc': 'Keyword'}
    category_colors = {'loc': 'red', 'org': 'blue', 'per': 'green', 'misc': 'yellow'}

    # For each category, im getting top terms and their scores
    for cat in categories:
        top_terms_scores = get_top_tfidf_terms(entities_tfidf[cat], filtered_data, f'a_{cat}')
        top_entities[cat] = top_terms_scores  # List of (term, score)

    # Building the node list
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
            node_categories[term_normalized] = cat
            term_set.add(term_normalized)

    # Building the graph with weighted edges
    G = nx.Graph()
    for idx, node in enumerate(nodes):
        G.add_node(node, category=node_categories[node])

    # Initializing edge weights
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
        # Adding edges between all pairs of entities in this row
        for i in range(len(entities_in_row)):
            for j in range(i+1, len(entities_in_row)):
                e1 = entities_in_row[i]
                e2 = entities_in_row[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]['weight'] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    # Normalizing edge weights for layout algorithms
    weights = [d['weight'] for u, v, d in G.edges(data=True)]
    max_weight = max(weights) if weights else 1
    for u, v, d in G.edges(data=True):
        d['weight'] = d['weight'] / max_weight

    # Computing the layout using 'graphviz_layout' with 'neato'
    try:
        pos = graphviz_layout(G, prog='neato')
    except Exception as e:
        st.error(f"Error with Graphviz layout 'neato': {e}")
        st.warning("Falling back to spring layout.")
        pos = nx.spring_layout(G, weight='weight', seed=42)

    # Storing positions in node attributes for easy access
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # Preparing edge traces
    edge_x = []
    edge_y = []
    for u, v, d in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Preparing node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    # Normalizing node sizes for visualization
    max_size = max(node_sizes) if node_sizes else 1
    node_sizes_normalized = [(s / max_size) * 50 + 10 for s in node_sizes]  # Scale sizes between 10 and 60

    # Creating node trace
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

    # Computing degree for each node
    degrees = dict(G.degree(weight='weight'))
    nx.set_node_attributes(G, degrees, 'degree')

    # Computing clusters using the Louvain method
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'cluster')

    # Computing modularity
    modularity = community_louvain.modularity(partition, G)
    st.session_state['modularity'] = modularity

    # Collecting node data
    node_data = []
    for node in G.nodes(data=True):
        node_name = node[0]
        attrs = node[1]
        node_data.append({
            'node': node_name,
            'category': node_categories.get(node_name, 'unknown'),
            'size': attrs.get('degree', 0),
            'cluster': attrs.get('cluster', -1),
            'tfidf_score': next((score for term, score in top_entities.get(attrs['category'], []) if term.lower().strip() == node_name), 0),
        })

    # Collecting edge data
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'weight': d.get('weight', 1),
        })

    # Preparing dataframes for export
    nodes_df = pd.DataFrame(node_data)
    edges_df = pd.DataFrame(edge_data)

    if 'network_nodes' not in st.session_state:
        st.session_state['network_nodes'] = {}
    if 'network_edges' not in st.session_state:
        st.session_state['network_edges'] = {}

    st.session_state['network_nodes'][time_period_label] = nodes_df
    st.session_state['network_edges'][time_period_label] = edges_df

    network_bg_color = 'rgba(255, 255, 255, 0.4)'

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='Key Associations Network',
                        font=dict(
                            size=16,
                            color='#023059'
                        )
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    plot_bgcolor=network_bg_color,  
                    paper_bgcolor=network_bg_color,  
                    annotations=[],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
               )

    st.plotly_chart(fig, use_container_width=True)
    

def plot_frequency_over_time(term, category):
    yearly_data = data[data[category].str.contains(term, regex=False, na=False)]
    yearly_counts = yearly_data.groupby(yearly_data['year']).size()
    total_counts = data.groupby('year').size()
    
    yearly_frequencies = (yearly_counts / total_counts) * 100
    
    fig = px.bar(yearly_frequencies, labels={'value': '% of Entries', 'year': 'Year'},
                 title=f'Frequency of "{term}" Over Time in {category}')
    st.plotly_chart(fig, use_container_width=True)

def display_qa_pairs(year_data):
    st.markdown("<h3 style='color:#023059; font-weight:bold;'>Question and Answer Pairs with Sentiment Score</h3>", unsafe_allow_html=True)

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
    
    # Applying custom styling using pandas Styler
    styled_df = display_df.style.set_properties(
        **{
            'background-color': '#023059',  
            'color': 'white',
            'border-color': 'white',
            'font-weight': 'bold'
        }
    ).set_table_styles([
        {
            'selector': 'th',
            'props': [('background-color', '#023059'), ('color', 'white'), ('font-weight', 'bold')]
        },
        {
            'selector': 'td',
            'props': [('background-color', '#023059'), ('color', 'white')]
        }
    ])
    
    st.dataframe(styled_df, key="qa_pairs")
    
    return display_df



def display_main_visualization(filtered_data, overall_data, selected_terms):
    """
    Displays the main visualization components including search term statistics,
    network visualization, time slider, timeline bar chart, helper text, Q&A pairs,
    and the download button.

    Parameters:
    - filtered_data (pd.DataFrame): The data filtered based on search terms.
    - overall_data (pd.DataFrame): The complete dataset for overall analysis.
    - selected_terms (list): List of search terms selected by the user.
    """
    
    # Displaying search term statistics above the main visualization
    st.markdown("<h3 style='color:#F29A2E;'>Search Term Statistics</h3>", unsafe_allow_html=True)
    
    # Defining maximum number of boxes per row
    max_boxes_per_row = 5
    
    if selected_terms:
        # Splitting selected_terms into chunks of max_boxes_per_row
        for i in range(0, len(selected_terms), max_boxes_per_row):
            chunk = selected_terms[i:i + max_boxes_per_row]
            cols = st.columns(len(chunk))
            for idx, term in enumerate(chunk):
                with cols[idx]:
                    processed_term = term.lower()
                    found = False
                    for category in ['a_per', 'a_loc', 'a_org', 'a_misc']:
                        category_stats = precomputed_stats.get(category, {})
                        term_stats = category_stats.get(processed_term)
                        if term_stats:
                            rank = term_stats['rank']
                            category_name = {
                                "a_per": "Person",
                                "a_loc": "Location",
                                "a_org": "Organization",
                                "a_misc": "Keyword"
                            }[category]
                            # Building the message with adjusted font sizes and dark blue background
                            message = f"""
                            <div style="
                                background-color: #023059;
                                color: white;
                                padding: 10px;
                                margin-bottom: 10px;
                                border-radius: 5px;
                                border: 2px solid #023059;
                                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
                            ">
                                <p style="font-size:14px; font-weight:bold; margin:0;">{term.capitalize()}</p>
                                <p style="font-size:12px; font-weight:bold; margin:0;">#{rank} {category_name}</p>
                                <p style="font-size:10px; font-weight:normal; margin:0;">most commonly mentioned by the Chinese Foreign Ministry</p>
                            </div>
                            """
                            st.markdown(message, unsafe_allow_html=True)
                            found = True
                            break
                    if not found:
                        st.markdown(f"""
                        <div style="
                            background-color: #023059;
                            color: white;
                            padding: 10px;
                            margin-bottom: 10px;
                            border-radius: 5px;
                            border: 2px solid #023059;
                            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
                        ">
                            <p style="font-size:14px; font-weight:bold; margin:0;">{term.capitalize()}</p>
                            <p style="font-size:12px; font-weight:bold; margin:0;">No Rank Available</p>
                            <p style="font-size:10px; font-weight:normal; margin:0;">Not found in any category</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    
    if filtered_data.empty:
        st.warning("No data available for the selected criteria.")
        return
    
    time_granularity = st.session_state.get('time_granularity', 'Yearly')
    
    time_filtered_data = pd.DataFrame()
    time_period_label = "Undefined"
    
    if time_granularity == 'Yearly':
        try:
            min_year = int(filtered_data['year'].min())
            max_year = int(filtered_data['year'].max())
        except KeyError:
            st.error("The 'year' column is missing from the filtered data.")
            return
        except ValueError:
            st.error("The 'year' column contains non-integer values.")
            return

        if min_year < max_year:
            selected_range = st.slider(
                "x",
                label_visibility='collapsed',
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="time_slider_yearly"
            )
            # Filtering data based on selected range
            time_filtered_data = filtered_data[
                (filtered_data['year'] >= selected_range[0]) & (filtered_data['year'] <= selected_range[1])
            ]
            time_period_label = f"{selected_range[0]} to {selected_range[1]}"
        else:
            st.info(f"Only data for the year {min_year} is available.")
            selected_range = (min_year, max_year)
            time_filtered_data = filtered_data.copy()
            time_period_label = f"{selected_range[0]}"
    
    else:
        # Monthly Granularity
        try:
            if not pd.api.types.is_datetime64_any_dtype(filtered_data['date']):
                filtered_data['date'] = pd.to_datetime(filtered_data['date'], errors='coerce')
            if not pd.api.types.is_datetime64_any_dtype(overall_data['date']):
                overall_data['date'] = pd.to_datetime(overall_data['date'], errors='coerce')
        except KeyError:
            st.error("The 'date' column is missing from the data.")
            return

        # Creating a new column 'month_period' as datetime objects
        filtered_data.loc[:, 'month_period'] = filtered_data['date'].dt.to_period('M').dt.to_timestamp()
        overall_data.loc[:, 'month_period'] = overall_data['date'].dt.to_period('M').dt.to_timestamp()

        min_date = filtered_data['month_period'].min()
        max_date = filtered_data['month_period'].max()

        # Convertiung to Python datetime objects
        if isinstance(min_date, pd.Timestamp):
            min_date = min_date.to_pydatetime()
        if isinstance(max_date, pd.Timestamp):
            max_date = max_date.to_pydatetime()
        
        if min_date < max_date:
            selected_range = st.slider(
                "x",
                label_visibility='collapsed',
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="MMM YYYY",
                key="time_slider_monthly"
            )
            # Filtering data based on selected range
            time_filtered_data = filtered_data[
                (filtered_data['month_period'] >= selected_range[0]) & (filtered_data['month_period'] <= selected_range[1])
            ]
            time_period_label = f"{selected_range[0].strftime('%Y-%m')} to {selected_range[1].strftime('%Y-%m')}"
        else:
            # Handling the case where min_date == max_date
            st.info(f"Only data for {min_date.strftime('%Y-%m')} is available.")
            selected_range = (min_date, max_date)
            time_filtered_data = filtered_data.copy()
            time_period_label = f"{selected_range[0].strftime('%Y-%m')}"
    
    if 'time_filtered_data' not in locals():
        st.error("An unexpected error occurred: 'time_filtered_data' is not defined.")
        time_filtered_data = pd.DataFrame()  
        time_period_label = "Undefined"
    
    if not time_filtered_data.empty:
        display_top_entities_tfidf(time_filtered_data, time_period_label)
        plot_colored_timeline(time_filtered_data, overall_data, time_granularity)

        col_toggle, col_spacer, col_help = st.columns([1.3, 0.1, 3.8])
        with col_toggle:
            def update_granularity():
                st.session_state['time_granularity'] = 'Monthly' if st.session_state['monthly_granularity_toggle'] else 'Yearly'
                st.session_state['granularity_changed'] = True

            monthly_granularity_toggle = st.checkbox(
                label='Use Monthly Granularity',
                value=(time_granularity == 'Monthly'),
                key='monthly_granularity_toggle',
                help="Toggle to switch between monthly and yearly data aggregation. 'Monthly' provides more granular insights.",
                on_change=update_granularity
            )
            
        st.session_state['time_granularity'] = 'Monthly' if monthly_granularity_toggle else 'Yearly'
            
        col_spacer.markdown("")
        col_help.markdown("<p style='color:#023059; font-size:15px; font-style: italic; font-weight: bold; margin: 0;'>Bar colors show if the Chinese MFA spoke more positively or negatively about your search terms compared to their average tone that year.</p>", unsafe_allow_html=True)
        
        # Displaying Q&A pairs across the full width
        display_df = display_qa_pairs(time_filtered_data)
        
        report_entry = {
            'selected_terms': selected_terms,
            'logic_type': st.session_state['logic_type_toggle'] and 'AND' or 'OR',
            'time_granularity': st.session_state['time_granularity'],
            'date_range': selected_range,
            'network_nodes': st.session_state['network_nodes'].get(time_period_label, pd.DataFrame()),
            'network_edges': st.session_state['network_edges'].get(time_period_label, pd.DataFrame()),
            'tfidf_df': st.session_state.get('tfidf_df', None),
            'qa_pairs': display_df,
        }
        
        st.session_state['report_data'].append(report_entry)
        # Enforcing here maximum number of report entries
        if len(st.session_state['report_data']) > MAX_REPORT_ENTRIES:
            st.session_state['report_data'] = st.session_state['report_data'][-MAX_REPORT_ENTRIES:]
        
        # Applying custom CSS styling
        st.markdown(
            """
            <style>
            /* Apply styles to all download buttons */
            .stDownloadButton > button {
                background-color: #023059 !important;
                color: white !important;
                font-weight: bold !important;
                font-size: 12px !important;
                height: 2em !important;
                width: auto !important;
                border-radius: 10px !important;
                border: 2px solid #023059 !important;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2) !important;
                margin-top: 10px !important;
            }
            .stDownloadButton > button:hover {
                background-color: #F29A2E !important;
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Generating the report content
        report_content = generate_report_content()
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"AIES_China_Dashboard_{timestamp}_Report.txt"

        st.markdown(
            """
            <style>
            /* Apply styles to the download button */
            .custom-download-button > button {
                background-color: #023059 !important; /* Custom Dark Blue */
                color: white !important;
                font-weight: bold !important;
                font-size: 12px !important;
                height: 3em !important;
                width: 100% !important;
                border-radius: 8px !important;
                border: none !important;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2) !important;
                cursor: pointer;
            }
            .custom-download-button > button:hover {
                background-color: #F29A2E !important; /* Orange on Hover */
            }
            </style>
            """,
            unsafe_allow_html=True
        )


        with bottom():
            st.markdown('<div class="custom-download-button">', unsafe_allow_html=True)
            st.download_button(
                label="Download Enriched Data & Custom Report for Chatbot Interaction",
                data=report_content,
                file_name=file_name,
                mime='text/plain',
                # help="Download the custom data report to converse with your preferred chatbot. The report includes prompts to help analyze trends and generate insights on China's foreign policy discourse.",
                key='download_report_bottom'
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("No data available to display visualizations.")



def plot_colored_timeline(filtered_data, overall_data, time_granularity):
    # Ensuring both DataFrames are explicit copies..
    filtered_data = filtered_data.copy()
    overall_data = overall_data.copy()
    
    if time_granularity == 'Monthly':
        # Assigning to a new column 'period_monthly' as strings to avoid dtype conflicts
        filtered_data.loc[:, 'period_monthly'] = filtered_data['date'].dt.to_period('M').astype(str)
        overall_data.loc[:, 'period_monthly'] = overall_data['date'].dt.to_period('M').astype(str)
        
        # Using the new 'period_monthly' column for grouping
        period_col = 'period_monthly'
    else:
        # Assigning to a new column 'period_yearly' as strings for consistency
        filtered_data.loc[:, 'period_yearly'] = filtered_data['year'].astype(str)
        overall_data.loc[:, 'period_yearly'] = overall_data['year'].astype(str)
        
        # Using the new 'period_yearly' column for grouping
        period_col = 'period_yearly'
    
    # Aggregating counts and sentiments
    timeline_data = filtered_data.groupby(period_col).size().reset_index(name="Counts")
    sentiment_data = filtered_data.groupby(period_col)['a_sentiment'].mean().reset_index(name='Sentiment')
    overall_sentiment = overall_data.groupby(period_col)['a_sentiment'].mean().reset_index(name='Overall_Sentiment')
    
    # Merging dataframes
    timeline_data = timeline_data.merge(sentiment_data, on=period_col, how='left')
    timeline_data = timeline_data.merge(overall_sentiment, on=period_col, how='left')
    
    # Calculating sentiment deviation
    timeline_data['Sentiment_Deviation'] = timeline_data['Sentiment'] - timeline_data['Overall_Sentiment']
    
    # Assigning colors based on deviation
    def assign_color(dev):
        if dev <= -0.1:
            return '#ff0000'  # Red
        elif -0.1 < dev <= -0.05:
            return '#ffa700'  # Orange
        elif -0.05 < dev <= 0.05:
            return '#fff400'  # Yellow
        elif 0.05 < dev <= 0.1:
            return '#a3ff00'  # Light Green
        else:
            return '#2cba00'  # Dark Green
    
    timeline_data['Color'] = timeline_data['Sentiment_Deviation'].apply(assign_color)
    
    timeline_bg_color = 'rgba(255, 255, 255, 0.4)'
    
    # Creating the histogram
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=timeline_data[period_col],
        y=timeline_data["Counts"],
        marker_color=timeline_data['Color'],
        name="Entry Counts"
    ))
    
    fig.update_layout(
        title=dict(
            text='Mentions (and Emotions) Over Time',
            font=dict(
                size=18,
                color='#023059'  # Custom dark blue
            )
        ),
        xaxis_title='Period' if time_granularity == 'Monthly' else 'Year',
        yaxis_title='Counts',
        width=800,
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),  
        plot_bgcolor=timeline_bg_color,  
        paper_bgcolor=timeline_bg_color  
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

def generate_report_content():

    content = ""
    system_prompt = """
    You are an expert analyst in Chinese foreign policy with a deep understanding of the role of language, narratives, and strategic communication in international relations. You have been provided with data from a dashboard that analyzes press conferences by the Chinese Ministry of Foreign Affairs (MFA), covering over two decades of diplomatic discourse.

    **Your tasks are:**

    1. **Examine Strategic Narratives:**
       - Identify and interpret the key narratives employed by the Chinese MFA.
       - Discuss how these narratives construct China's identity and interests on the international stage.
       - Analyze how the narratives have evolved over time, particularly between different leadership eras (e.g., Hu Jintao and Xi Jinping).

    2. **Analyze Entity Clusters and Relationships:**
       - Examine how entities (locations, organizations, people, keywords) are connected and clustered in the network.
       - Interpret why certain entities are closely linked based on historical events, geopolitical strategies, or diplomatic relations.
       - Consider the significance of entity clusters in the context of China's foreign policy objectives.

    3. **Explore Relational Contexts in Q&A Pairs:**
       - Use the provided question-answer pairs to understand the contexts in which entities are mentioned together.
       - Identify patterns, recurring themes, and the use of language that reflects strategic messaging.
       - Analyze how the MFA's responses align with China's broader foreign policy narratives and strategies.

    4. **Interpret TF-IDF Scores and Key Terms:**
       - Analyze the significance of key terms with high TF-IDF scores.
       - Explain how these terms relate to the overarching narratives and policy positions.
       - Discuss changes in term importance over time and their correlation with major geopolitical events or shifts in policy.

    5. **Analyze Changes Over Time:**
       - Compare the entity networks across different time periods.
       - Identify how relationships between entities have evolved.
       - Discuss possible reasons for these changes based on historical events or shifts in policy.

    6. **Provide Insights and Draw Conclusions:**
       - Synthesize your analysis to provide meaningful insights into China's foreign policy discourse.
       - Discuss notable patterns, shifts, or anomalies, and what they reveal about China's international relations strategies.
       - Offer potential implications for global politics and suggest areas for further research.

    **Guidelines:**

    - **Support all interpretations and statements with empirical data from the provided file.** This includes statistical data, network property measures, sentiment scores, or verbatim quotes.
    - **When directly quoting the Chinese MFA's answers, always add the exact date(s) of the quote in brackets.**
    - **Contextualize** your analysis within the broader scope of international relations and political communication.
    - **Reference historical events**, policy changes, or international incidents that may explain the observed patterns.
    - **Consider the interplay** between domestic and international audiences in the MFA's messaging.
    - **Reflect on the use of language** as a tool for constructing social reality and influencing international perceptions.

    **Data:**
    """

    content += system_prompt

    for idx, entry in enumerate(st.session_state['report_data'], 1):
        content += f"\n\n# Report Entry {idx}\n"
        content += f"## Query Parameters:\n"
        content += f"- Selected Search Terms: {entry['selected_terms']}\n"
        content += f"- Logic Type: {entry['logic_type']}\n"
        content += f"- Time Granularity: {entry['time_granularity']}\n"
        content += f"- Date Range: {entry['date_range']}\n"

        # Including Network Analysis
        content += "\n## Network Analysis:\n"
        nodes_df = entry['network_nodes']
        edges_df = entry['network_edges']

        if not nodes_df.empty and not edges_df.empty:
            content += f"### Nodes:\n{nodes_df.to_csv(index=False)}\n"
            content += f"### Edges:\n{edges_df.to_csv(index=False)}\n"
        else:
            content += "No network data available.\n"

        # Including TF-IDF Results
        tfidf_df = entry['tfidf_df']
        if tfidf_df is not None:
            content += "\n## TF-IDF Results:\n"
            for column in tfidf_df.columns:
                if column == 'index':
                    continue
                content += f"\n### {column}:\n"
                top_terms = tfidf_df[column].dropna().sort_values(ascending=False).head(10)
                for term, score in top_terms.items():
                    content += f"- {term}: {score}\n"
        else:
            content += "No TF-IDF results available.\n"

        # Including Q&A Pairs
        qa_pairs = entry['qa_pairs']
        content += f"\n## Question-Answer Pairs:\n{qa_pairs.to_csv(index=False)}\n"

    return content

with st.sidebar:
    
    st.markdown(
        """
        <style>
        .center-image {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px; /* Adjust this value to change the gap */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Applying the custom class to the image
    st.markdown(
        '<div class="center-image"><img src="https://www.aies.at/img/layout/AIES-Logo-EN-white.png?m=1684934843" width="180"></div>',
        unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
        .compact-text {
            font-size: 0.85em;
            line-height: 1.2;
            margin-bottom: 0.9em;
        }
        .compact-text h4 {
            margin-bottom: 0.2em;
        }
        .compact-text p {
            margin-bottom: 0.3em;
        }
        .feature-list {
            font-size: 1.1em;
            padding-left: 0.5em;
        }
        .feature-list div {
            margin-bottom: 0.2em;
        }
    </style>
    <div class="compact-text">
        <p>🔎 Explore China's Foreign Policy Statements</p>
        <p>Dive into 20+ years of press conferences by the Chinese Foreign Ministry and:</p>
        <div class="feature-list">
            <div>🔄 <strong>search</strong> for key terms in the MFA's statements.</div>
            <div>🕸️ <strong>visualize</strong> networks of associated entities.</div>
            <div>📊 <strong>analyze</strong> sentiment trends and narratives.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    with st.form(key='search_form'):
        selected_terms = st.multiselect("Select Search Criteria:", all_search_values, key='select_terms', help="Type in the places, topics, persons or organizations you are interested in and click 'Analyze'.")
        submitted = st.form_submit_button("Analyze")
        logic_type_toggle = st.toggle(
            label='Use AND Logic',
            value=False,  # Default is 'OR'
            key='logic_type_toggle',
            help="Toggle to switch between 'OR' and 'AND' logic for filtering. 'OR' shows more results: e.g., selecting 'USA' and 'China' will show statements mentioning either country. 'AND' is more restrictive: it will only show statements mentioning both countries."
        )

        # Mapping the toggle state to logic type
        logic_type = 'AND' if logic_type_toggle else 'OR'
        
    if submitted:
        st.session_state['filtered_data'] = filter_data(selected_terms, logic_type)
        
    
    # CSS for button styling
    st.markdown(
        """
        <style>
                .compact-text .attribution {
            margin-top: 0.7em;
            font-size: 14px;
        }
        div.stButton > button:first-child {
            background-color: #023059; /* Dark Blue */
            color: white;
            font-weight: bold;
            font-size: 12px;
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

    report_content = generate_report_content()
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"AIES_China_Dashboard_{timestamp}_Report.txt"

    report_bytes = report_content.encode('utf-8')
    
    st.download_button(
        label="Download Custom Data Analysis for LLM",
        data=report_bytes,
        file_name=file_name,
        mime='text/plain',
        )

    
    st.markdown(
        """
        <style>
            .compact-text {
                font-size: 0.8em;
                line-height: 1.2;
            }
            .compact-text p {
                margin-top: 0.2em;
                margin-bottom: 0.2em;
            }
            .attribution {
                font-size: 0.75em;
                margin-top: 0.5em;
                margin-bottom: 0.5em;
            }
            .citation {
                font-size: 1.0 em;
                font-style: italic;
                margin-top: 0.5em;
            }
        </style>
        <div class="compact-text">
            <p class="attribution">
                A big thank you to <a href="https://scholar.google.com/citations?user=dvrRIhAAAAAJ&hl=en" target="_blank">Richard Turcsányi</a> for his input! Original dataset can be found <a href="https://doi.org/10.1007/s11366-021-09762-3" target="_blank">here</a>.
            </p>
            <div style="height: 15px;"></div>
            <p class="citation">
                How to cite: <a href="https://www.linkedin.com/in/adamurosevic/" target="_blank">Urosevic, A.</a> (2024), Interactive China MFA Dashboard, Austrian Institute for European and Security Policy. Available at: https://www.aies.at/china-dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


filtered_data = filter_data(selected_terms, logic_type)

if filtered_data.empty:
    st.warning("No data available for the selected criteria.")


if st.session_state.filtered_data is not None:
    display_main_visualization(st.session_state.filtered_data, data, selected_terms)            
            
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

# Adding a spacer between buttons and the next section
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

if st.session_state.display_qa_pairs:
    # Adding an arrow and text to guide users to scroll down
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <span style='font-size: 24px;'>⬇️ Dive into the full Questions and Answers ⬇️</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    

data['date'] = pd.to_datetime(data['date'], errors='coerce')
