import csv
import json
data = []
with open("custom3_2.csv", 'r') as f:
    csvreader = csv.reader(f)
    for idx, row in enumerate(csvreader):
        try:
            data.append({"caption": row[0], "img_id": row[1]})
        except Error:
            continue
with open("custom3_2.json", "w") as f:
    json.dump(data, f)

