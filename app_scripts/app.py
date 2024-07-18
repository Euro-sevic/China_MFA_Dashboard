import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from scipy.sparse import load_npz
import pickle

st.set_page_config(layout="wide")

@st.cache_data
def load_data(dataset='full'):
    base_path = os.path.dirname(__file__)
    if dataset == 'full':
        file_path = os.path.join(base_path, '..', 'data', 'CMFA_PressCon_v4.xlsx')
    else:
        file_path = os.path.join(base_path, '..', 'data', 'ukraine_dataset_combined.xlsx')
        
    df = pd.read_excel(file_path)
    columns_to_clean = ["a_per", "a_loc", "a_org", "a_misc"]
    for column in columns_to_clean:
        df[column] = df[column].replace("-", np.nan).astype(str)
    return df


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
    base_path = os.path.dirname(__file__)
    # Correct the filename based on the slider input, which now corresponds to the file suffix.
    file_path = os.path.join(base_path, '..', 'data', f'yearly_tfidf_maxdf{tfidf_label}.pkl')
    with open(file_path, 'rb') as file:
        tfidf_data = pickle.load(file)
    return tfidf_data

def extract_relevant_tfidf(tfidf_data, filtered_data):
    yearly_documents = filtered_data.groupby('year')['answer_lem'].apply(lambda x: ' '.join(x.dropna())).to_dict()
    tfidf_scores = {year: {} for year in tfidf_data.keys()}
    for year, document in yearly_documents.items():
        if year in tfidf_data:
            matrix, feature_names = tfidf_data[year]
            feature_names_list = list(feature_names)
            document_terms = document.split()
            term_indices = [feature_names_list.index(term) for term in document_terms if term in feature_names_list]
            if term_indices:
                relevant_matrix = matrix[:, term_indices]
                summed_scores = np.array(relevant_matrix.sum(axis=0)).flatten()
                tfidf_scores[year] = dict(zip([feature_names_list[i] for i in term_indices], summed_scores))
    return pd.DataFrame(tfidf_scores)

def split_and_clean(text):
    return [
        t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"
    ]


def display_basic_stats():
    for category, terms in zip(
        ["a_per", "a_loc", "a_org", "a_misc"],
        [
            selected_people,
            selected_locations,
            selected_organizations,
            selected_miscellaneous,
        ],
    ):
        for term in terms:
            processed_term = term.lower()  # Convert term to lowercase to match dictionary keys
            category_stats = precomputed_stats.get(category, {})  # Safely get the category dictionary
            term_stats = category_stats.get(processed_term)  # Safely get the stats for the term

            if term_stats:
                rank = term_stats['rank']  # Access the rank
                with st.expander(f"🌟 {term.capitalize()} is the {rank}th most common value in {category}. 🌟"):
                    plot_frequency_over_time(term, category)
            else:
                st.error(f"Stats not found for '{term}' (processed as '{processed_term}')")


def plot_frequency_over_time(term, category):
    yearly_data = data[data[category].str.contains(term, regex=False, na=False)]
    yearly_counts = yearly_data.groupby(yearly_data['year']).size()
    total_counts = data.groupby('year').size()
    
    yearly_frequencies = (yearly_counts / total_counts) * 100
    
    fig = px.bar(yearly_frequencies, labels={'value': '% of Entries', 'year': 'Year'},
                 title=f'Frequency of "{term}" Over Time in {category}')
    st.plotly_chart(fig, use_container_width=True)


unique_people = sorted(set(item for sublist in data['a_per'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_organizations = sorted(set(item for sublist in data['a_org'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_locations = sorted(set(item for sublist in data['a_loc'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_miscellaneous = sorted(set(item for sublist in data['a_misc'].dropna().apply(split_and_clean).tolist() for item in sublist))

with st.sidebar:
    st.title("What does the official China say about...?")
    
    dataset_choice = st.radio(
        "Choose Dataset:",
        ('Full Dataset', 'Ukraine-only Dataset'),
        index=0
    )

    # Load the dataset based on user selection
    dataset = 'full' if dataset_choice == 'Full Dataset' else 'ukraine'
    data = load_data(dataset)
    
    st.markdown(
        """
    This interactive dashboard allows you to explore a corpus of the Chinese Ministry of Foreign Affairs press conferences. The dataset is a unique source of information for 20+ years of China's foreign policy discourse. Select different criteria to get insights from the data.
    """
    )

    st.title("Select Search Criteria")
    selected_locations = st.multiselect("Select Locations:", unique_locations, key='select_locations')
    selected_organizations = st.multiselect("Select Organizations:", unique_organizations, key='select_organizations')
    selected_people = st.multiselect("Select People:", unique_people, key='select_people')
    selected_miscellaneous = st.multiselect("Select other Keywords:", unique_miscellaneous, key='select_miscellaneous')
    logic_type = st.radio(
        "Select Logic Type:",
        ('AND', 'OR'),
        help="Choose 'AND' to display entries that meet all criteria or 'OR' for entries that meet any of the selected criteria."
    )


def filter_data(people, organizations, locations, miscellaneous, logic_type):
    conditions = []
    
    def build_condition(selected_items, column):
        if not selected_items:
            return None
        if logic_type == 'AND':
            return data[column].apply(lambda x: all(item in split_and_clean(x) for item in selected_items) if pd.notna(x) else False)
        else:
            return data[column].apply(lambda x: any(item in split_and_clean(x) for item in selected_items) if pd.notna(x) else False)

    people_condition = build_condition(people, 'a_per')
    organizations_condition = build_condition(organizations, 'a_org')
    locations_condition = build_condition(locations, 'a_loc')
    miscellaneous_condition = build_condition(miscellaneous, 'a_misc')
    
    valid_conditions = [cond for cond in [people_condition, organizations_condition, locations_condition, miscellaneous_condition] if cond is not None]
    
    if not valid_conditions:
        return data
    
    if logic_type == 'AND':
        return data[np.logical_and.reduce(valid_conditions)]
    else:
        return data[np.logical_or.reduce(valid_conditions)]


filtered_data = filter_data(selected_people, selected_organizations, selected_locations, selected_miscellaneous, logic_type)

def display_tfidf_scores():
    tfidf_data = load_tfidf_data()
    tfidf_df = extract_relevant_tfidf(tfidf_data, filtered_data)
    
    if not tfidf_df.empty:
        formatted_df = pd.DataFrame()

        for year in tfidf_df.columns:
            top_terms = tfidf_df[year].dropna().sort_values(ascending=False).head(10)
            if not top_terms.empty:
                max_score = top_terms.iloc[0]  # Get the maximum score to normalize
                formatted_terms = [f"{term} ({(score/max_score * 100):.2f}%)"
                                   for term, score in top_terms.items()]
                formatted_df[year] = pd.Series(formatted_terms).reset_index(drop=True)

        formatted_df = formatted_df.dropna(how='all', axis=1)

        if not formatted_df.empty:
            st.dataframe(formatted_df)
        else:
            st.write("All years resulted in empty data after processing.")
    else:
        st.write("No relevant TF-IDF scores found for the selected query.")




def display_top_entities(filtered_data):
    col1, col2, col3, col4 = st.columns(4)

    all_criteria = []
    if selected_locations:
        all_criteria.append(", ".join(selected_locations))
    if selected_organizations:
        all_criteria.append(", ".join(selected_organizations))
    if selected_people:
        all_criteria.append(", ".join(selected_people))
    if selected_miscellaneous:
        all_criteria.append(", ".join(selected_miscellaneous))
    combined_criteria = ", ".join(all_criteria)

    category_title = f"associated with '{combined_criteria}'" if all_criteria else "associated with selected criteria"

    with col1:
        st.subheader(f"Locations {category_title}")
        st.dataframe(filtered_data['a_loc'].apply(split_and_clean).explode().value_counts().sort_values(ascending=False))

    with col2:
        st.subheader(f"Organizations {category_title}")
        st.dataframe(filtered_data['a_org'].dropna().apply(split_and_clean).explode().value_counts().sort_values(ascending=False))

    with col3:
        st.subheader(f"People {category_title}")
        st.dataframe(filtered_data['a_per'].dropna().apply(split_and_clean).explode().value_counts().sort_values(ascending=False))

    with col4:
        st.subheader(f"Other Keywords {category_title}")
        st.dataframe(filtered_data['a_misc'].dropna().apply(split_and_clean).explode().value_counts().sort_values(ascending=False))


def plot_combined_timeline(filtered_data, overall_data):
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
        title_text="Sentiment and Entry Counts Over Time",
        xaxis_title="Year",
        yaxis_title="Counts",
        yaxis2_title="Average Sentiment",
        width=1200,
        height=600,
    )

    fig.update_yaxes(title_text="Entry Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=True)

    st.plotly_chart(fig, use_container_width=False)

display_basic_stats()

expander = st.expander("What & Who does China's MFA associate most commonly with your query?", expanded=True) 
with expander:
    display_top_entities(filtered_data)

expander_plot = st.expander("How Often and How Positively Does China's MFA Talk About This?", expanded=False)
with expander_plot:
    plot_combined_timeline(filtered_data, data)

def display_qa_pairs(year_data):
    st.write("Question and Answer Pairs with Sentiment Score", year_data[['question', 'answer', 'a_sentiment']])

years = filtered_data["year"].unique()
if len(years) > 1:
    selected_year = st.select_slider("Select a Year:", options=sorted(years))
    year_data = filtered_data[filtered_data["year"] == selected_year]
    if not year_data.empty:
        display_qa_pairs(year_data)
    else:
        st.error("No question-answer pairs to display for the selected year.")
elif len(years) == 1:
    selected_year = years[0]
    year_data = filtered_data[filtered_data["year"] == selected_year]
    st.info(f"Only data from the year {selected_year} is available based on your selections.")
    if not year_data.empty:
        display_qa_pairs(year_data)
    else:
        st.error("No question-answer pairs to display.")
else:
    st.error("No data available for the selected criteria.")

expander_tfidf = st.expander("Discover Influential Terms in MFA's Responses", expanded=False)
with expander_tfidf:
    # Use a select slider that outputs the numeric suffix directly
    tfidf_setting = st.select_slider(
        "Choose TF-IDF setting:",
        options=[50, 35, 20],  # These options directly correspond to the suffix in your file names
        format_func=lambda x: f"max_df={x/100:.2f}"  # Format display as decimal percentages
    )
    tfidf_data = load_tfidf_data(tfidf_setting)
    if st.button('Analyze Influential Terms'):
        filtered_data = filter_data(selected_people, selected_organizations, selected_locations, selected_miscellaneous, logic_type)
        tfidf_df = extract_relevant_tfidf(tfidf_data, filtered_data)
        if not tfidf_df.empty:
            formatted_df = pd.DataFrame()
            for year in tfidf_df.columns:
                top_terms = tfidf_df[year].dropna().sort_values(ascending=False).head(10)
                if not top_terms.empty:
                    max_score = top_terms.iloc[0]
                    formatted_terms = [f"{term} ({(score/max_score * 100):.2f}%)" for term, score in top_terms.items()]
                    formatted_df[year] = pd.Series(formatted_terms).reset_index(drop=True)
            formatted_df = formatted_df.dropna(how='all', axis=1)
            st.dataframe(formatted_df)
        else:
            st.write("No relevant TF-IDF scores found for the selected query.")
