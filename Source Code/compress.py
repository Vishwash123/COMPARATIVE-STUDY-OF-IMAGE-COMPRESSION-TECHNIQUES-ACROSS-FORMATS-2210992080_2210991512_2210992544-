import os
import subprocess

INPUT = "../dataset"
OUTPUT = "../output"

def run(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for file in os.listdir(INPUT):
    inp = os.path.join(INPUT, file)
    name = file.split('.')[0]

    run(["cwebp", "-q", "75", inp, "-o", f"{OUTPUT}/webp/{name}.webp"])
    run(["avifenc", "--min", "30", "--max", "30", inp, f"{OUTPUT}/avif/{name}.avif"])
    run(["cjxl", inp, f"{OUTPUT}/jxl/{name}.jxl", "-d", "1.5"])
    run(["ffmpeg", "-i", inp, "-q:v", "5", f"{OUTPUT}/jpeg/{name}.jpg"])

print("✅ Compression done")