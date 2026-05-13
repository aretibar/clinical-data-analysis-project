# Clinical Data Analysis Project

A Python-based clinical data analysis project simulating a small hypertension clinical trial dataset.

The project demonstrates:
- Clinical data cleaning
- Data quality checks
- Exploratory data analysis (EDA)
- Statistical hypothesis testing
- Clinical endpoint analysis
- Data visualization using Seaborn and Matplotlib

---

## Technologies Used

- Python
- Pandas
- Seaborn
- Matplotlib
- SciPy

---

## Clinical Dataset

The project uses a simulated dataset of 20 patients including:

- Age
- Sex
- Treatment group
- Baseline systolic blood pressure (SBP)
- Month 3 SBP
- Clinical response status

---

## Features

### Data Quality Checks
The project detects:
- Implausible ages
- Extreme blood pressure values
- Missing follow-up measurements

### Data Cleaning
Includes:
- Handling missing values
- Correcting unrealistic values
- Preparing clean datasets for analysis

### Statistical Analysis
The project performs:
- Independent t-test
- Chi-square test
- Group comparison analysis

### Data Visualization
Generated visualizations include:
- SBP change boxplots
- Responder rate barplots

---

## Example Outputs

Saved chart files:
- `sbp_change_boxplot.png`
- `responder_rate_barplot.png`

---

## How to Run

Install dependencies:

```bash
pip install pandas seaborn matplotlib scipy
