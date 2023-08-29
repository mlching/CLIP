#find how many times does a specific keyword exist in the training data pkl file
import argparse
import pickle
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to the pkl file')
    args = parser.parse_args()

    # Read the pkl file
    with open(args.file, 'rb') as f:
        data = pickle.load(f)
    
    while True:
        keyword = input("Enter keyword to search or type \"exit()\" to quit: ")
        if keyword == "exit()":
            break
        else:
            i = 0
            for d in data["captions"]:
                if keyword in d["caption"]:
                    i += 1

            print(f"Total number of \"{keyword}\" in {args.file} is {i}")

if __name__ == '__main__':
    main()

