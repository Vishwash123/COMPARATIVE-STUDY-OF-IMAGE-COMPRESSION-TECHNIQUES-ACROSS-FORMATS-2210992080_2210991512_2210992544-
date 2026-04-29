import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
import subprocess

INPUT = "../dataset"
OUTPUT = "../output"

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def get_size(path):
    return os.path.getsize(path) / 1024

def convert_to_png(input_path, output_path):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def butteraugli_score(orig_path, comp_path):
    temp_path = "../output/temp/temp.png"

    convert_to_png(comp_path, temp_path)

    try:
        result = subprocess.run(
            ["butteraugli", orig_path, temp_path, "diff.png"],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return None

data = []

for file in os.listdir(INPUT):
    orig_path = os.path.join(INPUT, file)
    orig = cv2.imread(orig_path)
    name = file.split('.')[0]

    formats = {
        "jpeg": ".jpg",
        "webp": ".webp",
        "avif": ".avif",
        "jxl": ".jxl"
    }

    for fmt, ext in formats.items():
        path = f"{OUTPUT}/{fmt}/{name}{ext}"

        if not os.path.exists(path):
            continue

        comp = cv2.imread(path)
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

df = pd.DataFrame(data, columns=[
    "Image", "Format", "PSNR", "SSIM", "Size(KB)", "Butteraugli"
])

df.to_csv("../results/results.csv", index=False)

print("✅ Metrics calculated")