import random
import pickle
import torch

number = 9

with open(f'./merge/split{number}.pkl', 'rb') as f:
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

j = 0
for idx, i in enumerate(shuffled_captions):
    i['clip_embedding'] = j
    shuffled_clip_embedding[idx] = shuffled_clip_embedding[idx].unsqueeze(0)
    j += 1
shuffled_data = {'clip_embedding': torch.cat(shuffled_clip_embedding, dim=0), 'captions': shuffled_captions}

print("all embedding: " ,len(shuffled_data['clip_embedding']))
print("all caption: " ,len(shuffled_data['captions']))
#with open('./data/shuffled_data8.pkl', 'wb') as f:
#    pickle.dump(shuffled_data, f)

index = int(0.95*len(data['captions']))
val_embedding =  shuffled_data['clip_embedding'][index:]
train_embedding = shuffled_data['clip_embedding'][:index]
val_captions = shuffled_data['captions'][index:]
train_captions = shuffled_data['captions'][:index]

j = 0
for idx, i in enumerate(train_captions):
    train_embedding[idx] = train_embedding[idx].unsqueeze(0)
    i['clip_embedding'] = j
    j += 1
j = 0
for idx, i in enumerate(val_captions):
    val_embedding[idx] = val_embedding[idx].unsqueeze(0)
    i['clip_embedding'] = j
    j += 1

val_data = {'clip_embedding':val_embedding, 'captions': val_captions}
train_data = {'clip_embedding': train_embedding, 'captions': train_captions}

print("train size: " ,len(train_data['captions']))
print("val size: ", len(val_data['captions']))
with open(f'./data/train_data{number}.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open(f'./data/val_data{number}.pkl', 'wb') as f:
    pickle.dump(val_data, f)

print("Done")
