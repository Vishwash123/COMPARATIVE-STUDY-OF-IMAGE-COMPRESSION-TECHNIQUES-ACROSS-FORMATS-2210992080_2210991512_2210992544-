import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/averages.csv")

# SSIM bar chart
plt.figure()
plt.bar(df.index, df["SSIM"])
plt.title("SSIM Comparison")
plt.savefig("../results/ssim.png")

# PSNR vs Size
plt.figure()
plt.scatter(df["Size(KB)"], df["PSNR"])

for i, txt in enumerate(df.index):
    plt.annotate(txt, (df["Size(KB)"][i], df["PSNR"][i]))

plt.xlabel("Size (KB)")
plt.ylabel("PSNR")
plt.title("PSNR vs Size")

plt.savefig("../results/psnr_vs_size.png")

print("✅ Graphs generated")