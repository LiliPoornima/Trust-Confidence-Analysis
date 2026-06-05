from scipy.stats import f_oneway, ttest_ind, pearsonr, shapiro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid")

# =========================================================
# Create Output Folders
# =========================================================

base_folder = "HCAHPS_Analysis/outputs"
fig_folder = os.path.join(base_folder, "figures")
data_folder = os.path.join(base_folder, "datasets")

os.makedirs(fig_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

# =========================================================
# Load Data
# =========================================================

state_results = pd.read_csv("HCAHPS_Analysis/data/state_results.csv")
measures = pd.read_csv("HCAHPS_Analysis/data/measures.csv")
states = pd.read_csv("HCAHPS_Analysis/data/states.csv")
reports = pd.read_csv("HCAHPS_Analysis/data/reports.csv")
responses = pd.read_csv("HCAHPS_Analysis/data/responses.csv")

# =========================================================
# Clean Column Names
# =========================================================

for df in [state_results, measures, states, reports, responses]:
    df.columns = df.columns.str.strip()

# =========================================================
# Merge Datasets
# =========================================================

df = pd.merge(state_results, measures, on="Measure ID", how="left")
df = pd.merge(df, states, on="State", how="left")
df = pd.merge(df, reports, on="Release Period", how="left")

# =========================================================
# Process Response Rate
# =========================================================

responses["Response Rate (%)"] = pd.to_numeric(
    responses["Response Rate (%)"], errors="coerce"
)

responses_grouped = responses.groupby(
    ["Release Period", "State"]
)["Response Rate (%)"].mean().reset_index()

df = pd.merge(
    df,
    responses_grouped,
    on=["Release Period", "State"],
    how="left"
)

df.to_csv(f"{data_folder}/combined_dataset.csv", index=False)

# =========================================================
# Create Trust and Confidence Scores
# =========================================================

trust_measures = [
    "Communication with Nurses",
    "Communication with Doctors",
    "Responsiveness of Hospital Staff"
]

confidence_measures = [
    "Overall Hospital Rating",
    "Willingness to Recommend the Hospital"
]

trust_df = df[df["Measure"].isin(trust_measures)]
confidence_df = df[df["Measure"].isin(confidence_measures)]

trust_score = trust_df.groupby(
    ["State", "Release Period", "Region"]
)["Top-box Percentage"].mean().reset_index()

trust_score.rename(columns={"Top-box Percentage": "Trust_Score"}, inplace=True)

confidence_score = confidence_df.groupby(
    ["State", "Release Period", "Region"]
)["Top-box Percentage"].mean().reset_index()

confidence_score.rename(
    columns={"Top-box Percentage": "Confidence_Score"}, inplace=True
)

final_df = pd.merge(
    trust_score,
    confidence_score,
    on=["State", "Release Period", "Region"]
)

if "Response Rate (%)" in df.columns:
    temp = df.groupby(["State", "Release Period"])[
        "Response Rate (%)"
    ].mean().reset_index()

    final_df = pd.merge(
        final_df,
        temp,
        on=["State", "Release Period"],
        how="left"
    )

final_df.to_csv(f"{data_folder}/final_analysis_dataset.csv", index=False)

# =========================================================
# Descriptive Statistics
# =========================================================

print("\nDescriptive Statistics")
print(final_df.describe())

# =========================================================
# Correlation Analysis
# =========================================================

corr, corr_p_value = pearsonr(
    final_df["Trust_Score"],
    final_df["Confidence_Score"]
)

print("\nCorrelation Analysis")
print("Correlation Coefficient:", round(corr, 4))
print("P-value:", round(corr_p_value, 6))

alpha = 0.05

print("\nHypothesis Testing (Alpha = 0.05)")
print("H0: No linear relationship between Trust and Confidence")
print("H1: Significant linear relationship exists")

if corr_p_value < alpha:
    print("Result: Reject H0 (Statistically Significant)")
else:
    print("Result: Fail to Reject H0 (Not Significant)")

# =========================================================
# T-Test
# =========================================================

high_trust = final_df[
    final_df["Trust_Score"] >= final_df["Trust_Score"].median()
]

low_trust = final_df[
    final_df["Trust_Score"] < final_df["Trust_Score"].median()
]

t_stat, t_p_value = ttest_ind(
    high_trust["Confidence_Score"],
    low_trust["Confidence_Score"]
)

print("\nT-Test Results")
print("T-stat:", round(t_stat, 4))
print("P-value:", round(t_p_value, 6))

# =========================================================
# ANOVA
# =========================================================

groups = [
    group["Confidence_Score"].values
    for name, group in final_df.groupby("Region")
]

f_stat, anova_p_value = f_oneway(*groups)

print("\nANOVA Results")
print("F-stat:", round(f_stat, 4))
print("P-value:", round(anova_p_value, 6))

# =========================================================
# Simple Linear Regression
# =========================================================

X = final_df[["Trust_Score"]]
y = final_df["Confidence_Score"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

# =========================================================
# Multiple Regression
# =========================================================

if "Response Rate (%)" in final_df.columns:
    final_df = final_df.dropna()

    X_multi = final_df[["Trust_Score", "Response Rate (%)"]]
    y_multi = final_df["Confidence_Score"]

    multi_model = LinearRegression()
    multi_model.fit(X_multi, y_multi)

    multi_pred = multi_model.predict(X_multi)

    print("\nMultiple Regression Coefficients")
    print("Trust Coefficient:", multi_model.coef_[0])
    print("Response Rate Coefficient:", multi_model.coef_[1])

# =========================================================
# Model Evaluation
# =========================================================

r2 = r2_score(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

print("\nModel Performance")
print("R2 Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))

# =========================================================
# Residual Analysis
# =========================================================

residuals = y - predictions

stat, shapiro_p_value = shapiro(residuals)

print("\nNormality Test (Shapiro-Wilk)")
print("Statistic:", round(stat, 4))
print("P-value:", round(shapiro_p_value, 6))

if shapiro_p_value > 0.05:
    print("Residuals are normally distributed")
else:
    print("Residuals are not normally distributed")

# =========================================================
# Small Contribution: Save Summary Results
# =========================================================

summary_results = pd.DataFrame({
    "Analysis": [
        "Correlation",
        "T-Test",
        "ANOVA",
        "Simple Linear Regression",
        "Shapiro-Wilk Normality Test"
    ],
    "Statistic": [
        corr,
        t_stat,
        f_stat,
        r2,
        stat
    ],
    "P-value": [
        corr_p_value,
        t_p_value,
        anova_p_value,
        None,
        shapiro_p_value
    ]
})

summary_results.to_csv(
    f"{data_folder}/summary_results.csv",
    index=False
)

print("\nSummary results saved successfully.")

# =========================================================
# Residual Scatter Plot
# =========================================================

plt.figure(figsize=(7, 5))
sns.scatterplot(x=predictions, y=residuals)

plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residual Plot (Model Fit Check)")
plt.xlabel("Predicted Confidence Score")
plt.ylabel("Residuals")

plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "residual_scatter.png"), dpi=150)
plt.close()

# =========================================================
# Visualizations
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(final_df["Trust_Score"], kde=True, ax=axes[0])
axes[0].set_title("Distribution of Trust Score")

sns.histplot(final_df["Confidence_Score"], kde=True, ax=axes[1])
axes[1].set_title("Distribution of Confidence Score")

plt.tight_layout()
plt.savefig(f"{fig_folder}/distributions.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=final_df[["Trust_Score", "Confidence_Score"]])

plt.title("Boxplot: Trust vs Confidence")
plt.tight_layout()
plt.savefig(f"{fig_folder}/boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))

heatmap_data = final_df[["Trust_Score", "Confidence_Score"]].copy()

if "Response Rate (%)" in final_df.columns:
    heatmap_data["Response Rate"] = final_df["Response Rate (%)"]

sns.heatmap(
    heatmap_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f"{fig_folder}/heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))

sns.regplot(
    x="Trust_Score",
    y="Confidence_Score",
    data=final_df,
    ci=95,
    scatter_kws={"s": 40}
)

plt.title("Trust vs Confidence (Regression)")
plt.tight_layout()
plt.savefig(f"{fig_folder}/regression.png", dpi=150, bbox_inches="tight")
plt.close()

final_df["Year"] = final_df["Release Period"].str[-4:]

trend = final_df.groupby("Year")[["Trust_Score", "Confidence_Score"]].mean()

plt.figure(figsize=(7, 4))
trend.plot()

plt.title("Trend of Trust & Confidence Over Time")
plt.tight_layout()
plt.savefig(f"{fig_folder}/trend.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=final_df,
    x="Trust_Score",
    y="Confidence_Score",
    hue="Region"
)

plt.title("State-Level Trust vs Confidence")
plt.tight_layout()
plt.savefig(f"{fig_folder}/state_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))

sns.pointplot(
    data=final_df,
    x="Region",
    y="Confidence_Score",
    errorbar="sd"
)

plt.title("Mean Confidence Score by Region (ANOVA Insight)")
plt.xlabel("Region")
plt.ylabel("Mean Confidence Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{fig_folder}/anova_mean_plot.png", dpi=150, bbox_inches="tight")
plt.close()

final_df["Trust_Group"] = np.where(
    final_df["Trust_Score"] >= final_df["Trust_Score"].median(),
    "High Trust",
    "Low Trust"
)

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=final_df,
    x="Trust_Group",
    y="Confidence_Score"
)

plt.title("T-Test Visualization: Confidence by Trust Group")
plt.xlabel("Trust Group")
plt.ylabel("Confidence Score")

plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "ttest_boxplot.png"), dpi=150)
plt.close()