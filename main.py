from master import MASTERModel
from custom_dataset import MultiIndexTimeSeriesDataset # Added import
import pickle
import numpy as np
import time

# Please install qlib first before load the data. # This comment might be outdated

# Remove or comment out the existing code block that loads dl_train, dl_valid, and dl_test from pickle files.
# universe = 'csi300' # ['csi300','csi800']
# prefix = 'opensource' # ['original','opensource'], which training data are you using
# train_data_dir = f'data'
# with open(f'{train_data_dir}\{prefix}\{universe}_dl_train.pkl', 'rb') as f:
#     dl_train = pickle.load(f)

# predict_data_dir = f'data\opensource'
# with open(f'{predict_data_dir}\{universe}_dl_valid.pkl', 'rb') as f:
#     dl_valid = pickle.load(f)
# with open(f'{predict_data_dir}\{universe}_dl_test.pkl', 'rb') as f:
#     dl_test = pickle.load(f)

# print("Data Loaded.")

csv_path = 'data/user_stock_data.csv'
lookback_window = 8

print("Loading data from CSV...")
# For now, use the same dataset for train, valid, and test to check pipeline integrity
# Proper data splitting (e.g., by time) should be implemented later.
dataset = MultiIndexTimeSeriesDataset(csv_path_or_buffer=csv_path, lookback_window=lookback_window) # Use csv_path_or_buffer
dl_train = dataset
dl_valid = dataset
dl_test = dataset
print(f"Data loaded. Number of sequences: {len(dataset)}")
if len(dataset) == 0:
    print("ERROR: No sequences were generated. Check CSV data and lookback window.")
    # Exit or raise an error if no data, as the model cannot run.
    raise SystemExit("Exiting due to no data.")


# d_feat is the number of features in each time step of a sequence
# Our dataset.__getitem__ returns (feature_sequence, label)
# feature_sequence is (lookback_window, num_actual_features)
# num_actual_features was 8 in the test of custom_dataset.py
if len(dataset.features) > 0:
    d_feat = dataset.features[0].shape[1]
else:
    # This case should be caught by the len(dataset) == 0 check above
    print("ERROR: Dataset is empty, cannot determine d_feat.")
    raise SystemExit("Exiting due to empty dataset.")
print(f"d_feat set to: {d_feat}")

d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
# Set gate_input_start_index = 0 and gate_input_end_index = d_feat
gate_input_start_index = 0
gate_input_end_index = d_feat 


# if universe == 'csi300': # beta can be set to a default or determined differently if universe is removed
#     beta = 5
# elif universe == 'csi800':
#     beta = 2
beta = 5 # Default beta, or adjust as needed

n_epoch = 1
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95


ic = []
icir = []
ric = []
ricir = []

# Training
######################################################################################
# Keep training loop commented out for now as per original, or enable one run
# For enabling one run for testing:
for seed in [0]: #  Run for one seed to test the pipeline
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model', save_prefix=f'user_data_model' # Adjusted save_prefix
    )

    start = time.time()
    # Train
    model.fit(dl_train, dl_valid)

    print("Model Trained.")

    # Test
    predictions, metrics = model.predict(dl_test)
    
    running_time = time.time()-start
    
    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
# End of the enabled training loop for one seed
######################################################################################

# Load and Test section - This section might fail if a pre-trained model for 'user_data_model_0.pkl' doesn't exist.
# For now, let's comment it out to focus on the data loading and training pipeline.
# If you want to test loading the model just trained, ensure save_prefix and param_path match.
######################################################################################
# for seed in [0]:
#     param_path = f'model/user_data_model_{seed}.pkl' # Adjusted path

#     print(f'Model Loaded from {param_path}')
#     model = MASTERModel(
#             d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
#             beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
#             n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
#             save_path='model/', save_prefix='user_data_model' # Adjusted save_prefix
#         )
#     try:
#         model.load_param(param_path)
#         predictions, metrics = model.predict(dl_test)
#         print(metrics)

#         ic.append(metrics['IC'])
#         icir.append(metrics['ICIR'])
#         ric.append(metrics['RIC'])
#         ricir.append(metrics['RICIR'])
#     except FileNotFoundError:
#         print(f"Model file {param_path} not found. Skipping load and test for this seed.")
    
######################################################################################

if ic: # Only print metrics if the ic list is not empty (i.e., training/testing was run)
    print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
    print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
    print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
    print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))
else:
    print("No metrics to report as training/testing was not fully completed or skipped.")
