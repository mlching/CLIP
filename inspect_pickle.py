import argparse
import pickle

# Create an argument parser
parser = argparse.ArgumentParser(description='Load and inspect pickle file')

# Add an argument for the path to the pickle file
parser.add_argument('pickle_file_path', type=str, help='Path to the pickle file')

# Parse the command-line arguments
args = parser.parse_args()

# Load the pickle file
with open(args.pickle_file_path, 'rb') as f:
    data = pickle.load(f)

# Inspect the loaded data
print(data)

