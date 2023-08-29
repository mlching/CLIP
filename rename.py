#rename all images in a folder
import os

folder_path = "./data_collected/custom2 - Copy"  # Set the path to the folder containing the images
i = 1  # Set the starting index for renaming

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):  # Check if the file is an image
        new_filename = f"{i}.jpg"  # Create a new filename with the index
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))  # Rename the file
        i += 1  # Increment the index for the next file
