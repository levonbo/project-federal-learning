import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv

df = pd.read_csv("results.csv", usecols=["params", "train_samples", "accuracy"])

pivot = df.pivot(index="train_samples", columns="params", values="accuracy")
pivot = pivot.sort_index(ascending=False)

sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Accuracy: Parameter Count vs. Training Samples")
plt.xlabel("Number of Parameters")
plt.ylabel("Amount trainingsdata")
plt.tight_layout()
plt.show()