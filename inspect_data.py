import pickle
import torch
import numpy as np
import pandas as pd

# Path to the data file
FILE_PATH = 'model/csi300_opensource_0.pkl'

def inspect_data(file_path):
    """
    Loads and inspects the data from a pickle file.
    """
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            loaded_object = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"\nType of loaded object: {type(loaded_object)}")

    if hasattr(loaded_object, 'get_index'):
        print("\nObject has a 'get_index' method.")
        try:
            index_output = loaded_object.get_index()
            print(f"First 10 elements of get_index() output: {index_output[:10]}")
            print(f"Total length of get_index() output: {len(index_output)}")
        except Exception as e:
            print(f"Error calling get_index(): {e}")
    else:
        print("\nObject does not have a 'get_index' method.")

    print("\nIterating through the loaded object (first 3 batches):")
    try:
        for i, batch in enumerate(loaded_object):
            if i >= 3:  # Limit to 3 batches
                break
            
            print(f"\n--- Batch {i} ---")
            print(f"Type of batch: {type(batch)}")

            if isinstance(batch, (torch.Tensor, np.ndarray)):
                print(f"Shape of batch: {batch.shape}")
                print(f"Dtype of batch: {batch.dtype}")
            elif isinstance(batch, (list, tuple)):
                print(f"Length of batch: {len(batch)}")
                if len(batch) > 0:
                    print(f"Type of first element: {type(batch[0])}")
                    if hasattr(batch[0], 'shape'):
                        print(f"Shape of first element: {batch[0].shape}")
                    elif hasattr(batch[0], 'size'):
                         print(f"Size of first element: {batch[0].size()}")
            # Add more specific checks if needed for other data types
            
    except TypeError:
        print("Loaded object is not iterable.")
    except Exception as e:
        print(f"Error during iteration: {e}")

if __name__ == '__main__':
    inspect_data(FILE_PATH)
