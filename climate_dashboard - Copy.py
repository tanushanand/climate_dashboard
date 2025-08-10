import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(page_title="Global Climate & CO₂ Dashboard", layout="wide")
st.title("🌍 Global Temperature & CO₂ Emissions Explorer")


@st.cache_data
def load_data():
    df = pd.read_csv("global_temp_income_co2_1960_2013.csv")
    return df

final_df = load_data()


st.sidebar.header("🔎 Filter Data")
countries = final_df["Country"].unique()
income_groups = final_df["Income group"].unique()
years = sorted(final_df["Year"].unique())

selected_countries = st.sidebar.multiselect("Select Country/Countries:", countries, default=["India", "United States"])
selected_income = st.sidebar.multiselect("Select Income Group(s):", income_groups, default=list(income_groups))
selected_years = st.sidebar.slider("Select Year Range:", min_value=min(years), max_value=max(years), value=(1960, 2013))


df_filtered = final_df[
    (final_df["Country"].isin(selected_countries)) &
    (final_df["Income group"].isin(selected_income)) &
    (final_df["Year"].between(selected_years[0], selected_years[1]))
]


st.markdown(f"### Showing data for **{len(df_filtered['Country'].unique())}** countries between **{selected_years[0]} - {selected_years[1]}**")


st.subheader("Temperature Trend Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_filtered, x="Year", y="MeanTemp", hue="Country", marker="o", ax=ax1)
ax1.set_ylabel("Mean Temperature (°C)")
ax1.set_title("Yearly Temperature by Country")
ax1.grid(True)
st.pyplot(fig1)


st.subheader("CO₂ Emissions vs Mean Temperature")
use_log = st.checkbox("Use log scale for CO₂", value=True)
fig2, ax2 = plt.subplots(figsize=(10, 5))
x = np.log10(df_filtered["CO2_kt"]) if use_log else df_filtered["CO2_kt"]
sns.scatterplot(x=x, y=df_filtered["MeanTemp"], hue=df_filtered["Country"], ax=ax2)
ax2.set_xlabel("CO₂ Emissions (kt)" + (" [log scale]" if use_log else ""))
ax2.set_ylabel("Mean Temperature (°C)")
ax2.set_title("CO₂ vs Temperature")
ax2.grid(True)
st.pyplot(fig2)


st.subheader("Distribution by Income Group")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**CO₂ Emissions by Income Group**")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_filtered, x="Income group", y="CO2_kt", ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)

with col2:
    st.markdown("**Mean Temperature by Income Group**")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_filtered, x="Income group", y="MeanTemp", ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    st.pyplot(fig4)
    

st.subheader("Average Temp & CO₂ Trends by Income Group")
grouped = df_filtered.groupby(["Year", "Income group"]).agg({"MeanTemp": "mean", "CO2_kt": "mean"}).reset_index()

fig5, ax5 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=grouped, x="Year", y="MeanTemp", hue="Income group", ax=ax5)
ax5.set_title("Average Temperature by Income Group")
ax5.set_ylabel("Temperature (°C)")
ax5.grid(True)
st.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=grouped, x="Year", y="CO2_kt", hue="Income group", ax=ax6)
ax6.set_title("Average CO₂ Emissions by Income Group")
ax6.set_ylabel("CO₂ Emissions (kt)")
ax6.grid(True)
st.pyplot(fig6)