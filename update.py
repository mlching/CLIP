import pickle
import torch

pickle_files = [
    "viznococoflickr.pkl",
    "custom1.pkl",
    "custom2.pkl"
]

merged_dict = {}
i = 0

for file_path in pickle_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        for key, value in data.items():
            if key in merged_dict:
                if isinstance(value, torch.Tensor):
                    merged_dict[key] = torch.cat((merged_dict[key], value), dim=0)
                elif isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                else:
                    print(f"Warning: Unable to merge key '{key}'. Incompatible value types.")
            else:
                merged_dict[key] = value

for j in merged_dict["captions"]:
    j["clip_embedding"] = i
    i += 1

with open("split7.pkl", "wb") as file:
    pickle.dump(merged_dict, file)

print("Merge completed successfully.")

# Print the keys
print("Keys in the merged dictionary:")
for key in merged_dict.keys():
    print(key)
