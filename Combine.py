# Combine all the processed files into one file

import glob

files = glob.glob("data/processed/*.txt")
combined = ""

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        combined += f.read() + "\n\n"

with open("data/combined.txt", "w", encoding="utf-8") as f:
    f.write(combined)
