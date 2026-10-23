import os
import shutil
import kagglehub

# Destination folder in your project
destination = "./data/animal-faces"
os.makedirs(destination, exist_ok=True)

# Download the dataset
dataset_path = kagglehub.dataset_download("andrewmvd/animal-faces")

# Copy all contents of the dataset folder to destination
for item in os.listdir(dataset_path):
    src = os.path.join(dataset_path, item)
    dst = os.path.join(destination, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

# Remove empty directories
for root, dirs, files in os.walk(destination, topdown=False):
    for d in dirs:
        dir_path = os.path.join(root, d)
        if not os.listdir(dir_path):  # directory is empty
            os.rmdir(dir_path)

# Optional: remove the cache folder after copying
# shutil.rmtree(dataset_path)

print("Dataset downloaded, copied, and empty folders removed in:", destination)
