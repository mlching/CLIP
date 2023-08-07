import torch

def read_pt_file(file_path):
    try:
        data = torch.load(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def main():
    file_name = input("Enter the path to the .pt file: ")

    data = read_pt_file(file_name)

    if data is not None:
        print(f"Successfully loaded the .pt file: {file_name}")
        while True:
            try:
                index = input("Enter the index to access the data (type 'exit' to quit): ")
                if index.lower() == 'exit':
                    break
                index = int(index)
                if 0 <= index < len(data['captions']):
                    print("Data at index",data['clip_embedding'][index])
                    print("Data at index",data['captions'][index])
                else:
                    print("Invalid index. Please enter a valid index between 0 and", len(data) - 1)
            except ValueError:
                print("Invalid input. Please enter a valid integer index or type 'exit' to quit.")

if __name__ == "__main__":
    main()

