import os
import pickle

folder='./data/conceptual/merge/'
db = {}
for filename in os.listdir(folder):
    if filename.endswith('.pkl'):
        myfile = open(folder+filename,"rb")
        db[os.path.splitext(filename)[0]] = pickle.load(myfile)
        myfile.close()
        print(filename)

myfile = open("./data/conceptual/merge/merge.pkl","wb")
pickle.dump(db, myfile)
myfile.close()