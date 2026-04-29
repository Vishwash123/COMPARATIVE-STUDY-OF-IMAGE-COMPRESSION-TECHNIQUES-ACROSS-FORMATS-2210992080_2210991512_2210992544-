import pandas as pd

df = pd.read_csv("../results/results.csv")

avg = df.groupby("Format").mean(numeric_only=True)

print(avg)

avg.to_csv("../results/averages.csv")

print("✅ Averages computed")