import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ssl

# Path to the JSON file containing image information
json_file = "./data/conceptual/cc12m_keywords.json"

# Directory to store the downloaded images
output_dir = "./cc12m_keywords"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Function to download an image
def download_image(image_info):
    image_id = image_info["img_id"]
    image_url = image_info["url"]

    # Extract the image filename from the URL
    image_filename = os.path.basename(image_url)

    # Build the output file path
    output_path = os.path.join(output_dir, f"{image_id}")

    # Download the image
    try:
        response = requests.get(image_url, stream=True, verify=False)
        #response.raise_for_status()
    except InsecureRequestWarning as e:
        i=0


    # Save the image to disk
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return image_id

# Create a ThreadPoolExecutor with a maximum number of threads
max_threads = 64  # Adjust the number of threads as per your system capabilities
executor = ThreadPoolExecutor(max_workers=max_threads)

# Iterate over the images and submit download tasks to the executor
futures = []
for image_info in data:
    future = executor.submit(download_image, image_info)
    futures.append(future)

# Show progress using tqdm
progress_bar = tqdm(total=len(futures), desc="Downloading images", unit="image")

# Process the completed download tasks
for future in as_completed(futures):
    image_id = future.result()
    progress_bar.update(1)
    progress_bar.set_postfix({"image_id": image_id})

# Wait for all tasks to complete
executor.shutdown()

progress_bar.close()
print("Download completed.")

