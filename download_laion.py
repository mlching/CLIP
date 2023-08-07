from datasets import load_dataset

# Load the dataset from Hugging Face Datasets Hub
dataset = load_dataset("laion/laion2B-en", "laion--laion2B-en")

# Convert the dataset to a list of dictionaries
data_list = dataset["train"].to_dict()

# Save the data as a JSON file
with open("dataset.json", "w") as json_file:
    json_file.write(json.dumps(data_list, indent=4))

print("Dataset downloaded and saved as 'dataset.json'.")



