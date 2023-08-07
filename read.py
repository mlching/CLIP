import argparse
import pickle
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to the pkl file')
    parser.add_argument('--index', required=True, type=int, help='Path to the pkl file')
    args = parser.parse_args()

    # Read the pkl file
    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    # Process the data or perform any operations you need
    # For demonstration purposes, we print the loaded data
    print(data["captions"][args.index])
    print(data["clip_embedding"][args.index])
    print("File caption length: ", len(data["captions"]))
    print("embedding length: ", len(data["clip_embedding"]))
if __name__ == '__main__':
    main()

