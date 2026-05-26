import zipfile
import os

zip_path = "fashionmnist.zip"
extract_dir = "dataset"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)

print("Extracted to:", extract_dir)