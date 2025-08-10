import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ============================
# Page Config
# ============================
st.set_page_config(page_title="Global Climate & CO₂ Dashboard", layout="wide")

# ============================
# Load Data
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("global_temp_income_co2_1960_2013.csv")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y").dt.year
    return df

final_df = load_data()

# ============================
# Sidebar Filters
# ============================
st.sidebar.header("Filter Options")
years = sorted(final_df["Year"].unique())
year_min, year_max = st.sidebar.select_slider(
    "Select year range:",
    options=years,
    value=(min(years), max(years))
)

all_countries = sorted(final_df["Country"].unique())
selected_countries = st.sidebar.multiselect(
    "Select countries:", all_countries, default=["India", "United States", "China"]
)

# Legend / plotting controls
st.sidebar.markdown("---")
st.sidebar.subheader("Legend / Display Options")
max_countries = st.sidebar.slider("Max countries to plot (by warming rate)", 3, 20, 8)
auto_aggregate = st.sidebar.checkbox("Auto-switch to Income Group view when too many countries", True)
show_line_labels = st.sidebar.checkbox("Label lines at right edge (hide legend)", True)

# ============================
# Filter Data
# ============================
df_filtered = final_df[
    (final_df["Year"] >= year_min) &
    (final_df["Year"] <= year_max) &
    (final_df["Country"].isin(selected_countries))
]

# ============================
# Rank by warming rate
# ============================
rank_src = (final_df
            .query("Year >= @year_min and Year <= @year_max")
            .sort_values(["Country","Year"])
            .assign(temp_change=lambda d: d.groupby("Country")["MeanTemp"].diff()))
rank = (rank_src.groupby("Country")["temp_change"]
        .mean().dropna().sort_values(ascending=False))

# ============================
# Temperature Trend Plot
# ============================
st.subheader("📊 Temperature Trend Over Time")
too_many = len(selected_countries) > max_countries
use_income_lines = auto_aggregate and too_many

fig1, ax1 = plt.subplots(figsize=(11, 4))
if use_income_lines:
    grouped_ct = (df_filtered.groupby(["Year","Income group"])["MeanTemp"]
                  .mean().reset_index())
    sns.lineplot(data=grouped_ct, x="Year", y="MeanTemp", hue="Income group", ax=ax1)
    ax1.set_title("Yearly Temperature by Income Group (auto-aggregated)")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Income group")
else:
    topN = [c for c in rank.index if c in selected_countries][:max_countries]
    plot_ct = df_filtered[df_filtered["Country"].isin(topN)]
    sns.lineplot(data=plot_ct, x="Year", y="MeanTemp", hue="Country", ax=ax1)
    ax1.set_title(f"Yearly Temperature by Country (Top {len(topN)} by warming rate)")
    if show_line_labels:
        leg = ax1.get_legend()
        if leg: leg.remove()
        for name, g in plot_ct.groupby("Country"):
            g = g.sort_values("Year")
            ax1.text(g["Year"].iloc[-1] + 0.2,
                     g["MeanTemp"].iloc[-1],
                     name, fontsize=8, va="center")
    else:
        ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Country", ncol=2)

ax1.set_ylabel("Mean Temperature (°C)")
ax1.grid(True)
fig1.tight_layout()
st.pyplot(fig1, clear_figure=True)

# ============================
# CO₂ vs Temperature Scatter
# ============================
st.subheader("🌡️ CO₂ Emissions vs Mean Temperature")
use_log = st.checkbox("Use log scale for CO₂ (safe log1p)", value=True)
plot_df = df_filtered.dropna(subset=["MeanTemp","CO2_kt"]).copy()
if use_log:
    plot_df["CO2_plot"] = np.log1p(plot_df["CO2_kt"])
    x_label = "CO₂ Emissions log1p(kt)"
else:
    plot_df["CO2_plot"] = plot_df["CO2_kt"]
    x_label = "CO₂ Emissions (kt)"

fig2, ax2 = plt.subplots(figsize=(11, 5))
if auto_aggregate and len(selected_countries) > max_countries:
    agg_sc = (plot_df.groupby(["Year","Income group"])
              .agg(MeanTemp=("MeanTemp","mean"),
                   CO2_plot=("CO2_plot","mean"))
              .reset_index())
    sns.scatterplot(data=agg_sc, x="CO2_plot", y="MeanTemp",
                    hue="Income group", ax=ax2, alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Income group")
else:
    topN = [c for c in rank.index if c in selected_countries][:max_countries]
    p = plot_df[plot_df["Country"].isin(topN)]
    sns.scatterplot(data=p, x="CO2_plot", y="MeanTemp",
                    hue="Country", ax=ax2, alpha=0.6)
    if show_line_labels:
        leg = ax2.get_legend()
        if leg: leg.remove()
    else:
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Country", ncol=2)

ax2.set_xlabel(x_label)
ax2.set_ylabel("Mean Temperature (°C)")
ax2.set_title("CO₂ vs Temperature")
ax2.grid(True)
fig2.tight_layout()
st.pyplot(fig2, clear_figure=True)

# ============================
# Regression Analysis
# ============================
st.subheader("📈 Regression Analysis: CO₂ → Temperature")
reg_df = df_filtered.dropna(subset=["MeanTemp","CO2_kt"])
if not reg_df.empty:
    X = reg_df[["CO2_kt"]]
    y = reg_df["MeanTemp"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.scatter(X, y, alpha=0.5, label="Data")
    ax3.plot(X, pred, color="red", linewidth=2, label="Regression Line")
    ax3.set_xlabel("CO₂ Emissions (kt)")
    ax3.set_ylabel("Mean Temperature (°C)")
    ax3.set_title("Linear Regression")
    ax3.legend()
    st.pyplot(fig3)
    st.write(f"**Regression Coefficient:** {model.coef_[0]:.4f}")
    st.write(f"**Intercept:** {model.intercept_:.4f}")
else:
    st.warning("Not enough data for regression in selected filters.")