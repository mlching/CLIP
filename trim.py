import pickle
import torch

pickle_files = [
    "./data/viz-wiz_val.pkl"
]

merged_dict = {}
i = 0

for file_path in pickle_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        merged_dict["clip_embedding"] = data["clip_embedding"][:10]
        merged_dict["captions"] = data["captions"][:10]
        print(len(merged_dict["clip_embedding"]))




with open("10val.pkl", "wb") as file:
    pickle.dump(merged_dict, file)

# Print the keys
print("Keys in the merged dictionary:")
for key in merged_dict.keys():
    print(key)
