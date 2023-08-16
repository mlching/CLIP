import csv
import json
data = []
with open("custom3.txt", 'r') as f:
    for idx, row in enumerate(f):
        try:
            caption, img_id = row.strip().split("\t")
        except ValueError:
            print(idx)
        data.append({"caption": caption, "img_id": img_id})
with open("custom3.json", "w") as f:
    json.dump(data, f)

