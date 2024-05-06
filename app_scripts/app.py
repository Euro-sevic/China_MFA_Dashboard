import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


@st.cache_data
def load_data():
    df = pd.read_excel("../data/CMFA_PressCon_v4.xlsx")
    columns_to_clean = ["a_per", "a_loc", "a_org", "a_misc"]
    for column in columns_to_clean:
        df[column] = df[column].replace("-", np.nan).astype(str)
    return df


data = load_data()


def split_and_clean(text):
    if pd.isna(text):
        return []
    return [
        item.strip()
        for item in str(text).split(";")
        if item.strip().lower() != "nan" and item.strip() != ""
    ]


def extract_unique_entries(column_name):
    unique_entries = set()

    def process_entry(entry):
        entries = entry.split(";")
        for part in entries:
            part = part.strip()
            if part.lower() != "nan" and part != "China" and part:
                if column_name == "a_loc" and "China-" in part:
                    subparts = part.split("China-")
                    unique_entries.update(
                        [
                            subpart.strip()
                            for subpart in subparts
                            if subpart.strip() and subpart.strip().lower() != "nan"
                        ]
                    )
                else:
                    unique_entries.add(part)

    data[column_name].dropna().explode().apply(process_entry)

    return sorted(unique_entries)


unique_people = extract_unique_entries("a_per")
unique_organizations = extract_unique_entries("a_org")
unique_locations = extract_unique_entries("a_loc")
unique_miscellaneous = extract_unique_entries("a_misc")

person = st.selectbox("Select a Person:", ["None"] + unique_people)
organization = st.selectbox("Select an Organization:", ["None"] + unique_organizations)
location = st.selectbox("Select a Location:", ["None"] + unique_locations)
miscellaneous = st.selectbox(
    "Select a Miscellaneous item:", ["None"] + unique_miscellaneous
)


def filter_data(df, person, organization, location):
    conditions = []

    if person != "None":
        conditions.append(df["a_per"].str.contains(person, na=False, regex=False))
    if organization != "None":
        conditions.append(df["a_org"].str.contains(organization, na=False, regex=False))
    if location != "None":
        conditions.append(df["a_loc"].str.contains(location, na=False, regex=False))

    if conditions:
        return df[np.logical_and.reduce(conditions)]
    else:
        return df


filtered_data = filter_data(data, person, organization, location)


def display_top_entities(filtered_data):
    df = filtered_data.copy()

    col1, col2 = st.columns(2)
    with col1:
        if "a_per" in df.columns:
            cleaned_people = (
                df["a_per"].dropna().apply(split_and_clean).explode().dropna()
            )
            top_people = cleaned_people.value_counts().sort_values(ascending=False)
            st.subheader("Associated Persons")
            st.dataframe(top_people)

    with col2:
        if "a_org" in df.columns:
            cleaned_orgs = (
                df["a_org"].dropna().apply(split_and_clean).explode().dropna()
            )
            top_organizations = cleaned_orgs.value_counts().sort_values(ascending=False)
            st.subheader("Associated Organizations")
            st.dataframe(top_organizations)

    col3, col4 = st.columns(2)
    with col3:
        if "a_loc" in df.columns:
            cleaned_locations = df["a_loc"].apply(split_and_clean).explode().dropna()
            top_locations = cleaned_locations.value_counts().sort_values(
                ascending=False
            )
            st.subheader("Associated Locations")
            st.dataframe(top_locations)

    with col4:
        if "a_misc" in df.columns:
            cleaned_misc = (
                df["a_misc"].dropna().apply(split_and_clean).explode().dropna()
            )
            top_misc = cleaned_misc.value_counts().sort_values(ascending=False)
            st.subheader("Associated Miscellaneous")
            st.dataframe(top_misc)


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
        title_text="Timeline and Sentiment Analysis",
        xaxis_title="Year",
        yaxis_title="Counts",
        yaxis2_title="Average Sentiment",
        width=1000,
        height=600,
    )

    fig.update_yaxes(title_text="Entry Count", secondary_y=False)
    fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=True)

    st.plotly_chart(fig, use_container_width=False)


filtered_data = filter_data(data, person, organization, location)
display_top_entities(filtered_data)
plot_combined_timeline(filtered_data, data)

years = filtered_data["year"].unique()
print("Unique years available:", years)

if len(years) > 1:
    selected_year = st.select_slider("Select a Year:", options=sorted(years))
    year_data = filtered_data[filtered_data["year"] == selected_year]
elif len(years) == 1:
    selected_year = years[0]
    year_data = filtered_data[filtered_data["year"] == selected_year]
    st.info(
        f"Only data from the year {selected_year} is available based on your selections."
    )
else:
    year_data = None
    st.error("No data available for the selected criteria.")

if year_data is not None and not year_data.empty:
    st.subheader(f"Question and Answer Pairs for {selected_year}")
    st.write(year_data[["question", "answer"]])
else:
    st.error("No question-answer pairs to display.")
