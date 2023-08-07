import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import struct

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    out_path = f"./data//viz-wiz_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/viz-wiz/annotations/val.json', 'r') as f:
        temp = json.load(f)
        train_data = temp["annotations"]
    print("%0d captions loaded from val.json " % len(train_data))
    all_embeddings = []
    all_captions = []
    j = 0
    for i in tqdm(range(len(train_data))):
        d = train_data[i]
        img_id = d["image_id"]
        filename = f"./data/viz-wiz/val/VizWiz_val_{int(img_id)-23431:08d}.jpg"
        try:
            image = io.imread(filename)
        except (ValueError, struct.error, SyntaxError, FileNotFoundError) as e:
            continue
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = j
        j += 0
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)



    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
