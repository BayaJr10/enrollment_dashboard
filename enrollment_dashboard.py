import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from io import BytesIO

# Set page config
st.set_page_config(page_title="Education Dashboard", page_icon="📊", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('student_enrollment_data_2014-2023.csv')
    # Convert Teacher_Ratio to numeric
    df['Teacher_Ratio'] = df['Teacher_Ratio'].str.split(':').apply(lambda x: int(x[1])/int(x[0]))
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)
selected_regions = st.sidebar.multiselect(
    "Select Region(s)",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Filter data
filtered_df = df[
    (df['Year'] >= selected_years[0]) & 
    (df['Year'] <= selected_years[1]) & 
    (df['Region'].isin(selected_regions))
]

# Dashboard title
st.title("📚 Student Enrollment Analytics Dashboard (2014-2023)")
st.markdown("""
Explore trends and relationships in student enrollment data across urban and rural regions.
""")

# Key Metrics
st.header("🔑 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Enrollment Rate", f"{filtered_df['Enrollment_Rate'].mean():.1f}%")
with col2:
    st.metric("Average Transition Rate", f"{filtered_df['Transition_Rate'].mean():.1f}%")
with col3:
    st.metric("Average Test Scores", f"{filtered_df['Test_Scores_Avg'].mean():.1f}")
with col4:
    st.metric("Urban-Rural Gap", 
              f"{(filtered_df[filtered_df['Region']=='Urban']['Enrollment_Rate'].mean() - filtered_df[filtered_df['Region']=='Rural']['Enrollment_Rate'].mean()):.1f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends Over Time", "🌍 Regional Comparison", "🔗 Relationships", "📊 Statistical Analysis"])

with tab1:
    st.header("Time Series Trends")
    trend_options = st.selectbox(
        "Select Metric to View",
        options=['Enrollment_Rate', 'Transition_Rate', 'Test_Scores_Avg', 
                'Internet_Access_Pct', 'Gov_Spending_PctGDP', 'Teacher_Ratio']
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=filtered_df, 
        x='Year', 
        y=trend_options, 
        hue='Region', 
        style='Region',
        markers=True,
        ax=ax
    )
    plt.title(f'{trend_options} Over Time')
    plt.ylabel(trend_options)
    st.pyplot(fig)
    
    # Crisis impact note
    if (2020 in filtered_df['Year'].unique()) or (2021 in filtered_df['Year'].unique()):
        st.info("Note: The dip in 2020-2021 shows the impact of the crisis period on education metrics.")

with tab2:
    st.header("Urban vs Rural Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        comp_metric = st.selectbox(
            "Select Metric for Comparison",
            options=['Enrollment_Rate', 'Transition_Rate', 'Test_Scores_Avg',
                    'Internet_Access_Pct', 'Gov_Spending_PctGDP', 'Teacher_Ratio',
                    'Distance_to_School', 'Poverty_Rate']
        )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=filtered_df,
        x='Region',
        y=comp_metric,
        ax=ax
    )
    plt.title(f'{comp_metric} Distribution by Region')
    st.pyplot(fig)
    
    # Show statistics
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.groupby('Region')[comp_metric].describe().style.format("{:.2f}"))

with tab3:
    st.header("Relationships Between Variables")
    
    relation = st.selectbox(
        "Select Relationship to Explore",
        options=[
            "Test Scores vs Transition Rate",
            "Birth Rate vs Enrollment Rate",
            "Government Spending vs Enrollment Rate",
            "Distance to School vs Enrollment Rate",
            "Internet Access vs Test Scores"
        ]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if relation == "Test Scores vs Transition Rate":
        sns.scatterplot(
            data=filtered_df,
            x='Test_Scores_Avg',
            y='Transition_Rate',
            hue='Region',
            style='Region',
            s=100,
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='Test_Scores_Avg',
            y='Transition_Rate',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--'},
            ax=ax
        )
        plt.title("Test Scores vs Transition Rate")
        plt.xlabel("Average Test Scores")
        plt.ylabel("Transition Rate (%)")
        
        # Calculate correlation
        corr = filtered_df[['Test_Scores_Avg', 'Transition_Rate']].corr().iloc[0,1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    elif relation == "Birth Rate vs Enrollment Rate":
        sns.scatterplot(
            data=filtered_df,
            x='Birth_Rate',
            y='Enrollment_Rate',
            hue='Region',
            style='Region',
            s=100,
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='Birth_Rate',
            y='Enrollment_Rate',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--'},
            ax=ax
        )
        plt.title("Birth Rate vs Enrollment Rate")
        plt.xlabel("Birth Rate (per 1000)")
        plt.ylabel("Enrollment Rate (%)")
        
        # Calculate correlation
        corr = filtered_df[['Birth_Rate', 'Enrollment_Rate']].corr().iloc[0,1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    elif relation == "Government Spending vs Enrollment Rate":
        sns.scatterplot(
            data=filtered_df,
            x='Gov_Spending_PctGDP',
            y='Enrollment_Rate',
            hue='Region',
            style='Region',
            s=100,
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='Gov_Spending_PctGDP',
            y='Enrollment_Rate',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--'},
            ax=ax
        )
        plt.title("Government Spending vs Enrollment Rate")
        plt.xlabel("Government Spending (% of GDP)")
        plt.ylabel("Enrollment Rate (%)")
        
        # Calculate correlation
        corr = filtered_df[['Gov_Spending_PctGDP', 'Enrollment_Rate']].corr().iloc[0,1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    elif relation == "Distance to School vs Enrollment Rate":
        sns.scatterplot(
            data=filtered_df,
            x='Distance_to_School',
            y='Enrollment_Rate',
            hue='Region',
            style='Region',
            s=100,
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='Distance_to_School',
            y='Enrollment_Rate',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--'},
            ax=ax
        )
        plt.title("Distance to School vs Enrollment Rate")
        plt.xlabel("Distance to School (km)")
        plt.ylabel("Enrollment Rate (%)")
        
        # Calculate correlation
        corr = filtered_df[['Distance_to_School', 'Enrollment_Rate']].corr().iloc[0,1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    elif relation == "Internet Access vs Test Scores":
        sns.scatterplot(
            data=filtered_df,
            x='Internet_Access_Pct',
            y='Test_Scores_Avg',
            hue='Region',
            style='Region',
            s=100,
            ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x='Internet_Access_Pct',
            y='Test_Scores_Avg',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--'},
            ax=ax
        )
        plt.title("Internet Access vs Test Scores")
        plt.xlabel("Internet Access (%)")
        plt.ylabel("Average Test Scores")
        
        # Calculate correlation
        corr = filtered_df[['Internet_Access_Pct', 'Test_Scores_Avg']].corr().iloc[0,1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    st.pyplot(fig)

with tab4:
    st.header("Statistical Analysis")
    
    st.subheader("Urban vs Rural Significance Tests")
    test_metric = st.selectbox(
        "Select Metric for Statistical Test",
        options=['Enrollment_Rate', 'Transition_Rate'])