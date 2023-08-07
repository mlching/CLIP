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
    out_path = f"./data/vg100k.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./region_descriptions.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from vg100k" % len(data))
    all_embeddings = []
    all_captions = []
    id_list = []
    counter = 0
    for i in range(len(data)):
        id_list.append(int(data[i]['id']))
    print("Number of images: ", len(id_list))
    for n, i in tqdm(enumerate(id_list)):
        filename = f"./vg100k/{int(i)}.jpg"
        try:
            image = io.imread(filename)
        except (ValueError, struct.error, SyntaxError, FileNotFoundError) as e:
            continue
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        for j in range(len(data[n]["regions"])):
            dict = {'caption': data[n]['regions'][j]['phrase'], 'clip_embedding': counter}
            all_captions.append(dict)
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
        counter += 1



    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print("%0d captions saved" % len(all_captions))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
