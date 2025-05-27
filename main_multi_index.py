import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Attempt to import from user-provided master.py and base_model.py
# These files should be in the same directory or Python path.
try:
    from master import MASTERModel
    from base_model import SequenceModel, DailyBatchSamplerRandom, zscore, calc_ic as base_calc_ic
    # calc_ic from base_model.py will be used by SequenceModel.predict
except ImportError as e:
    print(f"Error importing MASTERModel or SequenceModel: {e}")
    print("Please ensure master.py and base_model.py (with your provided code) are in the same directory.")
    raise

# Memory and Logging (placeholders for now)
LONG_SHORT_TERM_MEMORY_FILE = "memory_log.json"
LESSONS_LEARNED_FILE = "lessons_learned.md"

def log_to_memory(data):
    # Basic implementation, can be expanded
    print(f"Memory log: {data}")

def log_lesson(lesson):
    # Basic implementation, can be expanded
    print(f"Lesson learned: {lesson}")

def calculate_ic(predictions, actuals):
    """Calculates Spearman Rank Correlation (Information Coefficient)."""
    if len(predictions) < 2 or len(actuals) < 2:
        return 0.0, 0.0 # Not enough data for correlation
    
    # Ensure inputs are 1D arrays
    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()

    # Check for constant arrays which can cause issues with spearmanr
    if np.all(predictions == predictions[0]) or np.all(actuals == actuals[0]):
        log_to_memory({"event": "calculate_ic_constant_array_warning"})
        return 0.0, 1.0 # Return 0 correlation, p-value 1 to indicate issue

    try:
        ic_value, p_value = spearmanr(predictions, actuals)
        if np.isnan(ic_value): # spearmanr can return nan if std dev is zero
            log_to_memory({"event": "calculate_ic_nan_result"})
            return 0.0, 1.0
        return ic_value, p_value
    except Exception as e:
        log_lesson(f"Error calculating IC: {e}")
        return 0.0, 1.0

class StockDataset(Dataset):
    def __init__(self, data_tensor, index_df):
        """
        Args:
            data_tensor (torch.Tensor): Tensor of shape (num_samples, seq_len, num_features_incl_label)
                                       The label is expected to be at data_tensor[:, -1, -1].
            index_df (pd.DataFrame): DataFrame with 'datetime' and 'instrument' (ticker) for get_index().
                                     The 'datetime' should be the end date of the sequence.
        """
        self.data_tensor = data_tensor
        self.index_df = index_df.reset_index(drop=True) # Ensure simple integer index for __getitem__
        self.pd_index = pd.MultiIndex.from_frame(self.index_df[['datetime', 'instrument']])


    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]

    def get_index(self):
        # Required by DailyBatchSamplerRandom
        return self.pd_index


def load_and_preprocess_data(csv_path, sequence_len=60):
    log_to_memory({"event": "load_and_preprocess_data_start", "csv_path": csv_path})

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        log_lesson(f"File not found: {csv_path}. Ensure correct path.")
        print(f"Error: File not found at {csv_path}")
        return (None,) * 4 # Returning (datasets_tuple, d_feat_csv)

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    except ValueError as e:
        log_lesson(f"Date conversion error: {e}. Check date format in CSV. Expected YYYY-MM-DD.")
        print(f"Error converting date column: {e}. Please ensure dates are in YYYY-MM-DD format.")
        return (None,) * 4

    df = df.set_index(['ticker', 'date'])
    df = df.sort_index()

    df['returns'] = df.groupby(level='ticker')['closeadj'].pct_change().fillna(0)
    
    features_to_use = ['open', 'high', 'low', 'close', 'volume', 'closeadj']
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        log_lesson(f"Missing expected features: {missing_features}. Required: {features_to_use}")
        print(f"Error: Missing features in CSV: {missing_features}. Required: {features_to_use}")
        return (None,) * 4
        
    df_features = df[features_to_use].copy()
    df_target_raw = df[['returns']].copy() # Raw returns, SequenceModel will handle zscore

    filled_features_list = []
    for ticker_name, group in df_features.groupby(level='ticker'):
        filled_group = group.ffill().bfill()
        filled_features_list.append(filled_group)
    
    if not filled_features_list:
        df_features = pd.DataFrame(columns=features_to_use) 
    else:
        df_features = pd.concat(filled_features_list)

    df_features.dropna(how='any', inplace=True) # Drop rows with any NaNs in features
    
    common_index = df_features.index.intersection(df_target_raw.index)
    df_features = df_features.loc[common_index]
    df_target_raw = df_target_raw.loc[common_index]

    if df_features.empty:
        log_lesson("DataFrame became empty after handling missing values.")
        print("Error: DataFrame empty after handling missing values.")
        return (None,) * 4

    feature_scaler_map = {}
    scaled_features_list = []
    for ticker, group in df_features.groupby(level='ticker'):
        if not group.empty and not group.isnull().all().all():
            scaler = MinMaxScaler()
            scaled_group_features = scaler.fit_transform(group)
            scaled_features_list.append(pd.DataFrame(scaled_group_features, index=group.index, columns=group.columns))
            feature_scaler_map[ticker] = scaler
    
    if not scaled_features_list:
        log_lesson("No data available after attempting to scale features.")
        print("Error: No data to process after feature scaling step.")
        return (None,) * 4
        
    df_scaled_features = pd.concat(scaled_features_list)
    d_feat_csv = df_scaled_features.shape[1] # Number of primary features from CSV
    log_to_memory({"event": "features_scaled_per_ticker", "d_feat_csv": d_feat_csv})

    # Align target with scaled features
    df_target_aligned = df_target_raw.loc[df_scaled_features.index]

    # Prepare data for SequenceModel:
    # Each sample: (T, F_model_input + 1), where F_model_input = d_feat_csv * 2
    # Label is at sample[:, -1, -1]
    # Metadata: (end_date, ticker) for each sample
    
    all_sequences_data = []
    all_sequences_metadata_list = [] # List of dicts {'datetime': end_date, 'instrument': ticker}

    for ticker, group_data in df_scaled_features.groupby(level='ticker'):
        ticker_features_np = group_data.values # (num_days_for_ticker, d_feat_csv)
        # Ensure target is aligned with features for this ticker
        ticker_target_np = df_target_aligned.loc[group_data.index]['returns'].values # (num_days_for_ticker,)

        if len(ticker_features_np) >= sequence_len + 1: # Need one more for the label
            for i in range(len(ticker_features_np) - sequence_len):
                # X_primary: (sequence_len, d_feat_csv)
                X_primary_seq = ticker_features_np[i : i + sequence_len, :]
                
                # X_for_gate: (sequence_len, d_feat_csv) - duplicating primary for now
                X_gate_seq = X_primary_seq 
                
                # Concatenate for MASTER model input: (sequence_len, d_feat_csv * 2)
                X_model_input_seq = np.concatenate([X_primary_seq, X_gate_seq], axis=1)
                
                current_label = ticker_target_np[i + sequence_len] # Label corresponds to next step
                
                # Create combined tensor: (sequence_len, d_feat_csv * 2 + 1)
                # The last "feature column" will hold the label at its last time step
                combined_sample = np.zeros((sequence_len, d_feat_csv * 2 + 1), dtype=np.float32)
                combined_sample[:, :-1] = X_model_input_seq
                combined_sample[-1, -1] = current_label # Label at last time step of last feature column
                
                all_sequences_data.append(combined_sample)
                
                # Metadata: end date of sequence is date of label
                end_date_of_sequence = group_data.index[i + sequence_len][1] # date part of MultiIndex
                all_sequences_metadata_list.append({'datetime': end_date_of_sequence, 'instrument': ticker})

    if not all_sequences_data:
        log_lesson(f"Not enough data to create sequences of length {sequence_len}.")
        print(f"Error: Insufficient data to create sequences with length {sequence_len}.")
        return (None,) * 4

    all_sequences_tensor = torch.tensor(np.array(all_sequences_data), dtype=torch.float32)
    all_sequences_metadata_df = pd.DataFrame(all_sequences_metadata_list)
    
    log_to_memory({"event": "sequences_created_for_stockdataset", 
                   "num_samples": all_sequences_tensor.shape[0],
                   "d_feat_csv": d_feat_csv})

    # Temporal split
    # Sort metadata by date to ensure chronological split if not already
    all_sequences_metadata_df = all_sequences_metadata_df.sort_values(by='datetime').reset_index(drop=True)
    sorted_indices = all_sequences_metadata_df.index.to_numpy() # Use index of sorted metadata
    
    all_sequences_tensor = all_sequences_tensor[sorted_indices]


    num_samples = all_sequences_tensor.shape[0]
    train_size = int(num_samples * 0.7)
    val_size = int(num_samples * 0.15)

    # Slicing based on sorted order
    train_data = all_sequences_tensor[:train_size]
    train_meta = all_sequences_metadata_df.iloc[:train_size]

    val_data = all_sequences_tensor[train_size : train_size + val_size]
    val_meta = all_sequences_metadata_df.iloc[train_size : train_size + val_size]
    
    test_data = all_sequences_tensor[train_size + val_size :]
    test_meta = all_sequences_metadata_df.iloc[train_size + val_size :]

    log_to_memory({
        "event": "data_split_shapes",
        "train_data": train_data.shape, "val_data": val_data.shape, "test_data": test_data.shape,
    })

    if train_data.shape[0] == 0 or val_data.shape[0] == 0:
        log_lesson("Data splitting resulted in empty train or validation sets.")
        print("Error: Train or validation data splits are empty.")
        return (None,) * 4

    train_dataset = StockDataset(train_data, train_meta)
    val_dataset = StockDataset(val_data, val_meta)
    test_dataset = StockDataset(test_data, test_meta)
    
    # No explicit target scaler needed here, SequenceModel handles zscore

    return (train_dataset, val_dataset, test_dataset), d_feat_csv


def run_training(csv_path, sequence_len=60, d_model=256, t_nhead=4, s_nhead=2, 
                 dropout=0.1, beta=1.0, n_epochs=10, lr=1e-4, seed=0, 
                 save_path='model_output/', save_prefix='master_model', gpu_id=0):
    log_to_memory({"event": "run_training_start", "params": locals()})

    datasets_tuple, d_feat_csv = load_and_preprocess_data(csv_path, sequence_len)
    
    if datasets_tuple is None or d_feat_csv is None:
        log_lesson("Data loading/preprocessing failed. Aborting training.")
        print("Failed to load or preprocess data. Aborting training.")
        return

    dl_train, dl_valid, dl_test = datasets_tuple
    
    # Parameters for MASTERModel based on d_feat_csv and original MASTER logic
    master_d_feat_param = d_feat_csv # Primary features
    master_gate_input_start_index = d_feat_csv # Start of gate features in concatenated input
    master_gate_input_end_index = d_feat_csv * 2 # End of gate features

    log_to_memory({
        "event": "model_hyperparameters_for_master",
        "d_feat_model": master_d_feat_param, "d_model": d_model, "t_nhead": t_nhead, 
        "s_nhead": s_nhead, "dropout": dropout, "beta": beta,
        "gate_input_start_index": master_gate_input_start_index,
        "gate_input_end_index": master_gate_input_end_index,
        "n_epochs_arg": n_epochs, "lr_arg": lr, "seed": seed,
        "save_path": save_path, "save_prefix": save_prefix, "gpu_id": gpu_id
    })
    
    # Ensure save_path directory exists
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log_to_memory({"event": "created_save_path_directory", "path": save_path})


    # Instantiate MASTERModel (which inherits from SequenceModel)
    # SequenceModel handles device placement, optimizer, etc.
    model = MASTERModel(
        d_feat=master_d_feat_param,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=dropout,  # Assuming T_dropout_rate and S_dropout_rate are same
        S_dropout_rate=dropout,
        beta=beta,
        gate_input_start_index=master_gate_input_start_index,
        gate_input_end_index=master_gate_input_end_index,
        n_epochs=n_epochs,
        lr=lr,
        GPU=gpu_id, # For SequenceModel
        seed=seed,  # For SequenceModel
        train_stop_loss_thred=0.95, # Example from original main.py, adjust as needed
        save_path=save_path,      # For SequenceModel
        save_prefix=f'{save_prefix}_{seed}' # For SequenceModel, includes seed
    )
    
    log_to_memory({"event": "MASTERModel_initialized", "device": str(model.device)})

    if n_epochs <= 0:
        print(f"Warning: Number of training epochs is {n_epochs}. No training will occur.")
        log_to_memory({"event": "training_skipped_due_to_n_epochs", "n_epochs_value": n_epochs})
        if dl_test is not None and dl_test.__len__() > 0:
            print("Attempting to load a pre-trained model for testing if parameters allow...")
            # This would require a load_param call, which is usually handled by user or specific workflow
            # For now, if n_epochs is 0, we just don't train.
        else:
            print("No test data to evaluate.")
        return


    print("Starting model training...")
    model.fit(dl_train, dl_valid) # Uses SequenceModel's fit method
    log_to_memory({"event": "training_complete_via_model_fit"})
    
    print("\nStarting model testing...")
    if dl_test is not None and dl_test.__len__() > 0 :
        # To load the best model saved by fit, if applicable by SequenceModel's logic:
        # model.load_param(f'{save_path}/{save_prefix}_{seed}.pkl') # Or however SequenceModel saves/loads
        # The original SequenceModel saves if train_loss <= train_stop_loss_thred.
        # If you want to test the *last* state, no load is needed.
        # If you want to test the *best* saved state, ensure load_param is called.
        # For now, assume SequenceModel.fit leaves the model in its final trained state
        # or handles best model loading internally if a save occurred.

        predictions, metrics = model.predict(dl_test) # Uses SequenceModel's predict method
        
        print("\nTest Metrics:")
        if metrics:
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
            log_to_memory({"event": "test_phase_end", "metrics": metrics})
        else:
            print("No metrics returned from predict.")
            log_to_memory({"event": "test_phase_end", "status": "no_metrics"})
        
        # Example: Save predictions if needed
        # predictions.to_csv(f"{save_path}/predictions_{save_prefix}_{seed}.csv")
        # log_to_memory({"event": "predictions_saved"})

    else:
        print("No test data available to evaluate.")
        log_to_memory({"event": "test_phase_skipped_no_data"})
    
    print("\nTraining and testing script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MASTER model using original pipeline structure.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--sequence_len", type=int, default=60, help="Length of input sequences.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of model internal state (d_model in MASTER).") # original main.py: 256
    parser.add_argument("--t_nhead", type=int, default=4, help="Number of heads for temporal attention.")
    parser.add_argument("--s_nhead", type=int, default=2, help="Number of heads for spatial attention.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.") # original main.py: 0.5
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for feature gate.") # original main.py: 5 for csi300, 2 for csi800
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs.") # original main.py: 1
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.") # original main.py: 1e-5
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--save_path", type=str, default="model_artefacts/", help="Directory to save models and results.")
    parser.add_argument("--save_prefix", type=str, default="master_custom_data", help="Prefix for saved model files.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use if CUDA is available.")
    
    args = parser.parse_args()

    log_to_memory({"event": "main_script_start", "args": vars(args)})
    
    # Override some defaults with original paper's typical values if desired, or keep as CLI args
    # For example, if we assume 'csi300' like scenario from original main.py for some params:
    # args.d_model = 256
    # args.dropout = 0.5
    # args.beta = 5 # This was universe-dependent in original
    # args.n_epochs = 1 # Often very few for such models if data is large
    # args.lr = 1e-5

    run_training(
        csv_path=args.csv_path,
        sequence_len=args.sequence_len,
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        dropout=args.dropout,
        beta=args.beta,
        n_epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save_path,
        save_prefix=args.save_prefix,
        gpu_id=args.gpu_id
    )
    log_to_memory({"event": "main_script_end"})

    # TODOs based on original structure and current implementation:
    # 1. Stationarity checks (ADF, differencing) - currently skipped. Add if essential before scaling.
    # 2. Time-series cross-validation: Current split is a simple chronological percentage split.
    #    The original DailyBatchSamplerRandom processes day by day, which is good for TS.
    #    Consider if more sophisticated CV (like rolling origin) is needed for hyperparam tuning (not part of fit/predict).
    # 3. Feature Engineering: Current script uses basic OHLCV. The original MASTER (158 features) implies more extensive feature engineering.
    #    The current setup will use the available CSV features for `d_feat` and duplicate them for the gate.
    # 4. SAttention: Review if `DailyBatchSamplerRandom` logic (batching all stocks for a given day)
    #    is fully compatible with how SAttention expects to see inter-stock relations.
    #    The original Qlib data loaders might have handled this transparently. Current StockDataset + DailyBatchSampler
    #    will provide all sequences ending on a particular day as one batch to the model.
    # 5. Pytests for the new data loading and StockDataset.
    # 6. Memory logging and lessons learned to be filled more thoroughly.
    # 7. Ensure `master.py` and `base_model.py` are present with the code you provided.
# End of script marker 
