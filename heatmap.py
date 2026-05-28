import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv

with open("results.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        print(row)

# Deine Runs als DataFrame
df = pd.read_csv("results.csv")

print(type(df))

# Pivot für Heatmap
pivot = df.pivot(index="train_samples", columns="params", values="accuracy")
pivot = pivot.sort_index(ascending=False)

# Plot
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Accuracy: Parameter Count vs. Training Samples")
plt.xlabel("Number of Parameters")
plt.ylabel("Amount trainingsdata")
plt.tight_layout()
plt.show()