import json
from tqdm import tqdm


with open("sbu-captions-all.json", "r") as f:
    data = json.load(f)

es_count = 0
el_count = 0
all = []

for idx, i in tqdm(enumerate(data["captions"]), total=len(data["image_urls"])):
    if "escalator" in i:
        all.append({"url": data["image_urls"][idx], "caption": i, "img_id": data["image_urls"][idx].split("/")[-1]})
    if "elevator" in i:
        all.append({"url": data["image_urls"][idx], "caption": i, "img_id": data["image_urls"][idx].split("/")[-1]})
with open("sbu_lift.json", "w") as f:
    json.dump(all, f)


    