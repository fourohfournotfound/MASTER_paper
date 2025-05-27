import pickle
import torch
import pandas as pd
import numpy as np
import subprocess
import sys

# Attempt to import qlib, install if not found
try:
    import qlib
    print("Successfully imported qlib.")
except ImportError:
    print("qlib not found. Attempting to install pyqlib...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyqlib"])
        import qlib
        print("Successfully installed and imported qlib.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing pyqlib: {e}")
        print("Please ensure qlib is installed manually if the error persists.")
        sys.exit(1) # Exit if qlib installation fails
    except ImportError:
        print("Failed to import qlib even after attempting installation.")
        sys.exit(1)


# Path to the data file
FILE_PATH = 'model/csi300_opensource_0.pkl' # Adjusted path

def inspect_data(file_path):
    """
    Loads and inspects the data from a pickle file.
    """
    print(f"\nLoading data from: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            # Provide a dummy persistent_load function for pickle
            class DummyUnpickler(pickle.Unpickler):
                def persistent_load(self, pid):
                    print(f"Encountered persistent ID: {pid}. Returning None.")
                    return None
            loaded_object = DummyUnpickler(f).load()
            #loaded_object = pickle.load(f) # Original pickle.load
        print("Successfully loaded object with pickle.load (using DummyUnpickler).")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading pickle file with pickle.load: {e}")
        print("Attempting to load with torch.load as a fallback...")
        try:
            loaded_object = torch.load(file_path, map_location=torch.device('cpu')) # Added map_location
            print("Successfully loaded object with torch.load().")
        except Exception as e_torch:
            print(f"Error loading with torch.load(): {e_torch}")
            return

    print(f"\nType of loaded object: {type(loaded_object)}")

    if hasattr(loaded_object, 'get_index'):
        print("\nObject has a 'get_index' method.")
        try:
            index_output = loaded_object.get_index()
            print(f"Type of get_index() output: {type(index_output)}")
            print(f"First 10 elements of get_index() output: {index_output[:10]}")
            print(f"Total length of get_index() output: {len(index_output)}")
        except Exception as e:
            print(f"Error calling get_index(): {e}")
    else:
        print("\nObject does not have a 'get_index' method.")

    print("\nIterating through the loaded object (first 3 batches/elements):")
    
    # Check if the object is a Qlib dataset
    if isinstance(loaded_object, qlib.data.dataset.DatasetH):
        print("Object is a Qlib DatasetH. Attempting to access segments...")
        try:
            # Accessing data segments as Qlib datasets typically work
            for segment_name in ["train", "valid", "test"]:
                if hasattr(loaded_object, segment_name) and getattr(loaded_object, segment_name) is not None:
                    segment_data = getattr(loaded_object, segment_name)
                    print(f"\n--- Segment: {segment_name} ---")
                    print(f"Type of segment data: {type(segment_data)}")
                    if isinstance(segment_data, pd.DataFrame):
                        print(f"Shape of segment DataFrame: {segment_data.shape}")
                        print(f"Info of segment DataFrame:")
                        segment_data.info()
                        print(f"First 5 rows of segment DataFrame:\n{segment_data.head()}")
                    elif isinstance(segment_data, (list, tuple)) and len(segment_data) > 0:
                         print(f"Length of segment data: {len(segment_data)}")
                         print(f"Type of first element: {type(segment_data[0])}")
                         if hasattr(segment_data[0], 'shape'):
                            print(f"Shape of first element: {segment_data[0].shape}")
                    # Add more specific checks if needed
                else:
                    print(f"Segment {segment_name} not found or is None.")
            
            # If the dataset itself is iterable (e.g. for preparing data)
            # This part might need adjustment based on how Qlib DatasetH is typically used for batching
            if hasattr(loaded_object, 'prepare'):
                 print("\nDataset has a 'prepare' method. Example of preparing a segment (e.g., 'train'):")
                 # This is a conceptual example; actual usage might vary
                 # data_prepared = loaded_object.prepare("train", col_set="feature", data_key=qlib.data.dataset.provider.DataProxy.DK_L)
                 # print(f"Type of prepared data: {type(data_prepared)}")

        except Exception as e:
            print(f"Error accessing Qlib dataset segments: {e}")

    # Generic iteration for other types of objects
    else:
        try:
            iterator = iter(loaded_object)
            for i, batch in enumerate(iterator):
                if i >= 3:  # Limit to 3 batches/elements
                    break
                
                print(f"\n--- Batch/Element {i} ---")
                print(f"Type of batch/element: {type(batch)}")

                if isinstance(batch, (torch.Tensor, np.ndarray)):
                    print(f"Shape: {batch.shape}")
                    print(f"Dtype: {batch.dtype}")
                elif isinstance(batch, pd.DataFrame):
                    print(f"Shape: {batch.shape}")
                    print("Info:")
                    batch.info()
                elif isinstance(batch, pd.Series):
                    print(f"Shape: {batch.shape}")
                    print("Info:")
                    batch.info()
                elif isinstance(batch, (list, tuple)):
                    print(f"Length: {len(batch)}")
                    if len(batch) > 0:
                        first_elem = batch[0]
                        print(f"Type of first element: {type(first_elem)}")
                        if hasattr(first_elem, 'shape'):
                            print(f"Shape of first element: {first_elem.shape}")
                        elif hasattr(first_elem, 'size') and callable(getattr(first_elem, 'size')):
                             print(f"Size of first element: {first_elem.size()}")
                # Add more specific checks if needed for other data types
                
        except TypeError:
            print("Loaded object is not directly iterable or not a Qlib DatasetH.")
        except Exception as e:
            print(f"Error during iteration: {e}")

if __name__ == '__main__':
    inspect_data(FILE_PATH)
