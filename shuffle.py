import random
import pickle
import torch


with open('./merge/split7.pkl', 'rb') as f:
    data = pickle.load(f)


clip_embedding = data['clip_embedding']
captions = data['captions']

# Combine the two lists into a single list of tuples (tensor, caption_dict)
data_pairs = list(zip(clip_embedding, captions))

random.shuffle(data_pairs)

# Separate the shuffled data pairs back into clip_embedding and captions lists
shuffled_clip_embedding, shuffled_captions = zip(*data_pairs)
shuffled_clip_embedding = list(shuffled_clip_embedding)
shuffled_captions = list(shuffled_captions)
print(len(shuffled_clip_embedding))
print(len(shuffled_captions))
print(shuffled_clip_embedding[0])

j = 0
for idx, i in enumerate(shuffled_captions):
    i['clip_embedding'] = j
    shuffled_clip_embedding[idx] = shuffled_clip_embedding[idx].unsqueeze(0)
    j += 1
shuffled_data = {'clip_embedding': torch.cat(shuffled_clip_embedding, dim=0), 'captions': shuffled_captions}

print(len(shuffled_data['clip_embedding']))
print(len(shuffled_data['captions']))
with open('./data/shuffled_data7.pkl', 'wb') as f:
    pickle.dump(shuffled_data, f)

print("Done")
