import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



# Make plots show nicely
sns.set(style="whitegrid")

# 1. CREATE A SMALL CLINICAL DATASET
data = {
    "patient_id": range(1, 21),
    "age": [45, 52, 60, 38, 70, 55, 49, 63, 58, 41,
            67, 72, 36, 50, 59, 47, 53, 62, 44, 69],
    "sex": ["F", "M", "F", "F", "M", "M", "F", "M", "F", "M",
            "F", "M", "F", "M", "F", "F", "M", "M", "F", "M"],
    "treatment_group": ["Drug", "Drug", "Placebo", "Drug", "Placebo",
                        "Drug", "Placebo", "Drug", "Placebo", "Drug",
                        "Placebo", "Drug", "Placebo", "Drug", "Placebo",
                        "Drug", "Placebo", "Drug", "Placebo", "Drug"],
    "baseline_sbp": [150, 160, 155, 148, 170, 165, 152, 158, 162, 149,
                     168, 175, 145, 159, 161, 151, 157, 169, 153, 172],
    "month3_sbp": [135, 142, 150, 140, 168, 150, 148, 146, 160, 145,
                   165, 170, 143, 150, 158, 147, 152, 166, 149, 170],
    "response": ["Responder", "Responder", "Non-responder", "Responder",
                 "Non-responder", "Responder", "Responder", "Responder",
                 "Non-responder", "Responder", "Non-responder", "Responder",
                 "Responder", "Responder", "Non-responder", "Responder",
                 "Responder", "Non-responder", "Responder", "Non-responder"]
}

df = pd.DataFrame(data)

print("First 5 rows of the clinical dataset:")
print(df.head())

print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- SUMMARY STATISTICS ---")
print(df.describe())

print("\n--- VALUE COUNTS ---")
print("Treatment groups:")
print(df["treatment_group"].value_counts())
print("\nResponse:")
print(df["response"].value_counts())

# 5. INTRODUCE SOME DATA QUALITY ISSUES (SIMULATED)
df_dirty = df.copy()
df_dirty.loc[0, "age"] = 5       # impossible age
df_dirty.loc[4, "baseline_sbp"] = 300  # extremely high SBP
df_dirty.loc[10, "month3_sbp"] = None  # missing follow-up

print("\n--- DIRTY DATA EXAMPLES ---")
print(df_dirty.loc[[0, 4, 10], :])

# 6. SIMPLE DATA QUALITY CHECKS
print("\n--- DATA QUALITY CHECKS ---")
implausible_age = df_dirty[(df_dirty["age"] < 18) | (df_dirty["age"] > 100)]
print("Implausible ages:")
print(implausible_age)

implausible_sbp = df_dirty[(df_dirty["baseline_sbp"] < 70) | (df_dirty["baseline_sbp"] > 250)]
print("\nImplausible baseline SBP:")
print(implausible_sbp)

missing_month3 = df_dirty[df_dirty["month3_sbp"].isna()]
print("\nMissing month 3 SBP:")
print(missing_month3)

# 7. CLEAN THE DATA
df_clean = df_dirty.copy()

# Fix implausible age by setting to median age
median_age = df_clean["age"].median()
df_clean.loc[df_clean["age"] < 18, "age"] = median_age

# Cap extreme SBP values at 250
df_clean.loc[df_clean["baseline_sbp"] > 250, "baseline_sbp"] = 250

# Option 1: Drop rows with missing month3_sbp
df_clean = df_clean.dropna(subset=["month3_sbp"])

print("\n--- CLEANED DATA (rows 0, 4, 10) ---")
print(df_clean.loc[[0, 4], :])  # 10 will be dropped

# 8. DEFINE A CLINICAL ENDPOINT: CHANGE IN SBP
df_clean["sbp_change"] = df_clean["month3_sbp"] - df_clean["baseline_sbp"]

print("\n--- SBP CHANGE (first 10 patients) ---")
print(df_clean[["patient_id", "treatment_group", "baseline_sbp", "month3_sbp", "sbp_change"]].head(10))

# 9. EDA: SUMMARY BY TREATMENT GROUP
group_summary = df_clean.groupby("treatment_group")["sbp_change"].agg(["count", "mean", "std", "min", "max"])
print("\n--- SBP CHANGE BY TREATMENT GROUP ---")
print(group_summary)

# EDA: RESPONSE RATES BY TREATMENT GROUP
response_table = pd.crosstab(df_clean["treatment_group"], df_clean["response"])
response_rates = response_table.div(response_table.sum(axis=1), axis=0)

print("\n--- RESPONSE COUNTS BY TREATMENT GROUP ---")
print(response_table)
print("\n--- RESPONSE RATES BY TREATMENT GROUP ---")
print(response_rates)

# 10. VISUALIZATION: SBP CHANGE BY TREATMENT GROUP
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="treatment_group", y="sbp_change")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Change in Systolic BP from Baseline to Month 3 by Treatment")
plt.ylabel("SBP change (month3 - baseline)")
plt.xlabel("Treatment group")
plt.tight_layout()
plt.savefig("sbp_change_boxplot.png")
plt.show()

# 11. VISUALIZATION: RESPONSE RATE BY TREATMENT GROUP
response_counts = df_clean.groupby(["treatment_group", "response"]).size().reset_index(name="count")
total_counts = df_clean.groupby("treatment_group")["patient_id"].count().reset_index(name="total")
response_merged = pd.merge(response_counts, total_counts, on="treatment_group")
response_merged["rate"] = response_merged["count"] / response_merged["total"]

plt.figure(figsize=(6, 4))
sns.barplot(
    data=response_merged[response_merged["response"] == "Responder"],
    x="treatment_group",
    y="rate"
)
plt.ylim(0, 1)
plt.title("Responder Rate by Treatment Group")
plt.ylabel("Proportion of Responders")
plt.xlabel("Treatment group")
plt.tight_layout()
plt.savefig("responder_rate_barplot.png")
plt.show()

# 12. STATISTICAL TESTING

# Separate SBP change by treatment group
drug_change = df_clean[df_clean["treatment_group"] == "Drug"]["sbp_change"]
placebo_change = df_clean[df_clean["treatment_group"] == "Placebo"]["sbp_change"]

# 12a. Independent t-test for SBP change
t_stat, p_value_t = stats.ttest_ind(drug_change, placebo_change, equal_var=False)

print("\n--- T-TEST: SBP CHANGE (DRUG vs PLACEBO) ---")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value_t:.3f}")

# 12b. Chi-square test for responder rates
contingency_table = pd.crosstab(df_clean["treatment_group"], df_clean["response"])
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- CHI-SQUARE TEST: RESPONSE RATES (DRUG vs PLACEBO) ---")
print("Contingency table:")
print(contingency_table)
print(f"\nChi-square statistic: {chi2:.3f}")
print(f"p-value: {p_value_chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print("\nExpected counts under null hypothesis:")
print(expected)

# 13. SIMPLE INTERPRETATION (TEXT SUMMARY)

print("\n--- INTERPRETATION (SIMPLIFIED) ---")

mean_changes = df_clean.groupby("treatment_group")["sbp_change"].mean()
drug_mean = mean_changes["Drug"]
placebo_mean = mean_changes["Placebo"]

print(f"Mean SBP change (Drug): {drug_mean:.1f} mmHg")
print(f"Mean SBP change (Placebo): {placebo_mean:.1f} mmHg")
print(f"Difference (Drug - Placebo): {drug_mean - placebo_mean:.1f} mmHg")

if p_value_t < 0.05:
    print("The difference in SBP change between Drug and Placebo is statistically significant (p < 0.05).")
else:
    print("The difference in SBP change between Drug and Placebo is NOT statistically significant (p >= 0.05).")

responder_rates_simple = response_rates["Responder"]
print("\nResponder rate (Drug): {:.1%}".format(responder_rates_simple["Drug"]))
print("Responder rate (Placebo): {:.1%}".format(responder_rates_simple["Placebo"]))

if p_value_chi2 < 0.05:
    print("The difference in responder rates between Drug and Placebo is statistically significant (p < 0.05).")
else:
    print("The difference in responder rates between Drug and Placebo is NOT statistically significant (p >= 0.05).")




