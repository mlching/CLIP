import pickle
import torch


with open('./data/shuffled_data7.pkl', 'rb') as f:
    data = pickle.load(f)
index = int(0.85*len(data['captions']))
val_embedding =  data['clip_embedding'][index:]
train_embedding = data['clip_embedding'][:index]
val_captions = data['captions'][index:]
train_captions = data['captions'][:index]

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

print(len(train_data['captions']))
print(len(val_data['captions']))
with open('./data/train_data7.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('./data/val_data7.pkl', 'wb') as f:
    pickle.dump(val_data, f)

print("Done")

