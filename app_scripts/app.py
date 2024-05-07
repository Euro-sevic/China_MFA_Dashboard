import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide")
st.title("What does the official China say about...?")
st.markdown(
    """
This interactive dashboard allows you to explore a corpus of the Chinese Ministry of Foreign Affairs press conferences. The dataset is a unique source of information for 20+ years of China's foreign policy discourse. Select different criteria to get insights from the data.
"""
)


@st.cache_data
def load_data():
    df = pd.read_excel("../data/CMFA_PressCon_v4.xlsx")
    columns_to_clean = ["a_per", "a_loc", "a_org", "a_misc"]
    for column in columns_to_clean:
        df[column] = df[column].replace("-", np.nan).astype(str)
    return df


data = load_data()


def split_and_clean(text):
    return [
        t.strip() for t in text.split(";") if t.strip() and t.strip().lower() != "nan"
    ]

unique_people = sorted(set(item for sublist in data['a_per'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_organizations = sorted(set(item for sublist in data['a_org'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_locations = sorted(set(item for sublist in data['a_loc'].dropna().apply(split_and_clean).tolist() for item in sublist))
unique_miscellaneous = sorted(set(item for sublist in data['a_misc'].dropna().apply(split_and_clean).tolist() for item in sublist))

with st.expander("Select Search Criteria", expanded=True):
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
            # All selected items must match the record
            return data[column].apply(lambda x: all(item in split_and_clean(x) for item in selected_items) if pd.notna(x) else False)
        else:
            # Any selected items may match the record
            return data[column].apply(lambda x: any(item in split_and_clean(x) for item in selected_items) if pd.notna(x) else False)

    # Build conditions for each category
    people_condition = build_condition(people, 'a_per')
    organizations_condition = build_condition(organizations, 'a_org')
    locations_condition = build_condition(locations, 'a_loc')
    miscellaneous_condition = build_condition(miscellaneous, 'a_misc')
    
    # Collect all valid conditions
    valid_conditions = [cond for cond in [people_condition, organizations_condition, locations_condition, miscellaneous_condition] if cond is not None]
    
    if not valid_conditions:
        return data
    
    # Apply logical AND or OR across all valid conditions
    if logic_type == 'AND':
        return data[np.logical_and.reduce(valid_conditions)]
    else:
        return data[np.logical_or.reduce(valid_conditions)]



filtered_data = filter_data(selected_people, selected_organizations, selected_locations, selected_miscellaneous, logic_type)

def display_top_entities(filtered_data):
    col1, col2, col3, col4 = st.columns(4)

    # Combine all criteria into a single string
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

    # Use the combined criteria for all category titles
    category_title = f"Data associated with '{combined_criteria}'" if all_criteria else "Data associated with selected criteria"

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
        title_text="How Often and How Positively Does China's MFA Talk About This?",
        xaxis_title="Year",
        yaxis_title="Counts",
        yaxis2_title="Average Sentiment",
        width=1200,
        height=600,
    )

    fig.update_yaxes(title_text="Entry Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=True)

    st.plotly_chart(fig, use_container_width=False)


expander = st.expander("View Detailed Associations", expanded=True) 
with expander:
    display_top_entities(filtered_data)

plot_combined_timeline(filtered_data, data)

years = filtered_data["year"].unique()
if len(years) > 1:  
    selected_year = st.select_slider("Select a Year:", options=sorted(years))
    year_data = filtered_data[filtered_data["year"] == selected_year]
    if not year_data.empty:
        st.subheader(f"Question and Answer Pairs for {selected_year}")
        st.write(year_data[["question", "answer"]])
    else:
        st.error("No question-answer pairs to display for the selected year.")
elif len(years) == 1: 
    selected_year = years[0]
    year_data = filtered_data[filtered_data["year"] == selected_year]
    st.info(f"Only data from the year {selected_year} is available based on your selections.")
    if not year_data.empty:
        st.subheader(f"Question and Answer Pairs for {selected_year}")
        st.write(year_data[["question", "answer"]])
    else:
        st.error("No question-answer pairs to display.")
else:
    st.error("No data available for the selected criteria.")

