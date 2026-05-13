import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/sample_clinical_data.csv", sep=",")

print(df.columns)
print(df.head())

plt.figure(figsize=(8,5))

plt.plot(df["Patient_ID"], df["Cholesterol"], marker="o")

plt.xlabel("Patient ID")
plt.ylabel("Cholesterol")
plt.title("Cholesterol Levels by Patient")

plt.savefig("cholesterol_chart.png")

plt.show()