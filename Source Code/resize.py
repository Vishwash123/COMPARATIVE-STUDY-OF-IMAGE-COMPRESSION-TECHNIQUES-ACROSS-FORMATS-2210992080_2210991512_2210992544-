import cv2
import os

INPUT = "../dataset"
SIZE = (1024, 1024)

for file in os.listdir(INPUT):
    path = os.path.join(INPUT, file)
    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, SIZE)

    new_path = os.path.join(INPUT, file.split('.')[0] + ".png")
    cv2.imwrite(new_path, img)

    if path != new_path:
        os.remove(path)

print("✅ All images resized & converted to PNG")