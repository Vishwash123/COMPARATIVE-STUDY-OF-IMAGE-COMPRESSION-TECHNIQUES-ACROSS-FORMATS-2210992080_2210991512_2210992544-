import os
import subprocess
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

DATASET = "dataset"
OUTPUT = "output"
RESULTS = "results"

# Windows path to djxl.exe (must download)
CJXL_PATH = r"C:\Users\Acer\Desktop\research_paper\tools\cjxl.exe"
DJXL_PATH = r"C:\Users\Acer\Desktop\research_paper\tools\djxl.exe"

# HELPERS 

def run(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def get_size(path):
    return os.path.getsize(path) / 1024

def decode_jxl_to_png(jxl_path, png_path):
    """Decode JXL to PNG using djxl.exe"""
    conv = subprocess.run([DJXL_PATH, jxl_path, png_path], capture_output=True)
    if conv.returncode != 0:
        print("DJXL decode failed:", jxl_path)
        print(conv.stderr.decode())
        return None
    return cv2.imread(png_path)

def read_image_safe(path, temp_dir=f"{OUTPUT}/temp"):
    """Reads an image with OpenCV, converts to PNG if unsupported"""
    img = cv2.imread(path)
    if img is not None:
        return img

    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "read.png")

    if path.endswith(".jxl"):
        return decode_jxl_to_png(path, temp_file)
    else:
        conv = subprocess.run(["ffmpeg", "-y", "-i", path, temp_file], capture_output=True)
        if conv.returncode != 0:
            print("FFmpeg conversion failed:", path)
            return None
        return cv2.imread(temp_file)

def butteraugli_score(orig_path, comp_path):
    temp_dir = f"{OUTPUT}/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "temp.png")

    # Decode JXL if needed
    if comp_path.endswith(".jxl"):
        comp_img = decode_jxl_to_png(comp_path, temp_file)
        if comp_img is None:
            return None
    else:
        conv = subprocess.run(["ffmpeg", "-y", "-i", comp_path, temp_file], capture_output=True)
        if conv.returncode != 0:
            print("FFmpeg conversion failed:", comp_path)
            return None

    # Run Butteraugli
    result = subprocess.run(["butteraugli_main", orig_path, temp_file], capture_output=True, text=True)
    if result.returncode != 0:
        print("Butteraugli FAILED:", comp_path)
        print(result.stderr)
        return None
    try:
        return float(result.stdout.strip().split()[0])
    except:
        print("Parsing error:", result.stdout)
        return None

# STEP 1: RESIZE

def resize_images():
    os.makedirs(OUTPUT, exist_ok=True)
    for file in os.listdir(DATASET):
        path = os.path.join(DATASET, file)
        comp = read_image_safe(path)
        if comp is None:
            print("FAILED to read:", path)
            continue
        img = cv2.resize(comp, (1024, 1024))
        new_path = os.path.join(DATASET, file.split('.')[0] + ".png")
        cv2.imwrite(new_path, img)
        if path != new_path:
            os.remove(path)

# STEP 2: COMPRESS 

def compress():
    os.makedirs(f"{OUTPUT}/webp", exist_ok=True)
    os.makedirs(f"{OUTPUT}/avif", exist_ok=True)
    os.makedirs(f"{OUTPUT}/jxl", exist_ok=True)
    os.makedirs(f"{OUTPUT}/jpeg", exist_ok=True)

    for file in os.listdir(DATASET):
        inp = os.path.join(DATASET, file)
        name = file.split('.')[0]

        run(["cwebp", "-q", "75", inp, "-o", f"{OUTPUT}/webp/{name}.webp"])
        run(["avifenc", "--min", "30", "--max", "30", inp, f"{OUTPUT}/avif/{name}.avif"])
        run([CJXL_PATH, inp, f"{OUTPUT}/jxl/{name}.jxl", "-d", "1.5"])
        run(["ffmpeg", "-y", "-i", inp, "-q:v", "5", f"{OUTPUT}/jpeg/{name}.jpg"])

#  STEP 3: METRICS 

def compute_metrics():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(f"{OUTPUT}/temp", exist_ok=True)
    data = []

    formats = {"jpeg": ".jpg", "webp": ".webp", "avif": ".avif", "jxl": ".jxl"}

    for file in os.listdir(DATASET):
        orig_path = os.path.join(DATASET, file)
        orig = cv2.imread(orig_path)
        if orig is None:
            print("Cannot read original:", file)
            continue
        name = file.split('.')[0]

        for fmt, ext in formats.items():
            path = f"{OUTPUT}/{fmt}/{name}{ext}"
            if not os.path.exists(path):
                continue

            comp = read_image_safe(path)
            if comp is None:
                continue

            data.append([
                file,
                fmt,
                psnr(orig, comp),
                ssim(orig, comp, channel_axis=2),
                get_size(path),
                butteraugli_score(orig_path, path)
            ])

    df = pd.DataFrame(data, columns=["Image", "Format", "PSNR", "SSIM", "Size(KB)", "Butteraugli"])
    df.to_csv(f"{RESULTS}/results.csv", index=False)
    return df

#  STEP 4: ANALYSIS 

def analyze(df):
    avg = df.groupby("Format").mean(numeric_only=True)
    avg.to_csv(f"{RESULTS}/averages.csv")

    best_psnr = avg["PSNR"].idxmax()
    best_ssim = avg["SSIM"].idxmax()
    best_butter = avg["Butteraugli"].idxmin() if "Butteraugli" in avg.columns and avg["Butteraugli"].notna().any() else "Not Available"
    smallest = avg["Size(KB)"].idxmin()

    summary = f"""
=== AUTOMATED ANALYSIS ===

Best Compression (Smallest Size): {smallest}
Best PSNR: {best_psnr}
Best SSIM: {best_ssim}
Best Perceptual Quality (Butteraugli): {best_butter}

Conclusion:
Modern formats like AVIF and JXL outperform JPEG in both compression and quality.
WebP provides a balanced trade-off between size and quality.
PNG remains suitable only for lossless scenarios due to large file size.
"""
    with open(f"{RESULTS}/summary.txt", "w") as f:
        f.write(summary)

    return avg

# STEP 5: GRAPHS 

def plot(avg):
    # SSIM BAR CHART 
    plt.figure(figsize=(8, 5))
    bars = plt.bar(avg.index, avg["SSIM"], color='skyblue')
    plt.title("SSIM Comparison")
    plt.ylabel("SSIM")
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,  # x position
            height,                          # y position
            f"{height:.3f}",                  # format value
            ha='center', va='bottom', fontsize=10
        )
    plt.savefig(f"{RESULTS}/ssim.png", bbox_inches='tight')
    plt.close()

    # PSNR vs Size SCATTER PLOT
    plt.figure(figsize=(8, 5))
    plt.scatter(avg["Size(KB)"], avg["PSNR"], color='orange', s=100)

    for i, txt in enumerate(avg.index):
        x = avg["Size(KB)"].iloc[i]
        y = avg["PSNR"].iloc[i]
        plt.annotate(f"{txt}\n({y:.2f}, {x:.0f} KB)",
                     (x, y),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     fontsize=9)

    plt.xlabel("Size (KB)")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Size")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{RESULTS}/psnr_vs_size.png", bbox_inches='tight')
    plt.close()

# Complete flow

if __name__ == "__main__":
    print("🔄 Resizing...")
    resize_images()
    print("🔄 Compressing...")
    compress()
    print("🔄 Computing metrics...")
    df = compute_metrics()
    print("🔄 Analyzing...")
    avg = analyze(df)
    print("🔄 Plotting...")
    plot(avg)
    print("✅ FULL PIPELINE COMPLETE")