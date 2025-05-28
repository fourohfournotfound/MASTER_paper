import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import time
import datetime
import logging
import sys
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from master import MASTERModel
from base_model import SequenceModel, DailyBatchSamplerRandom, calc_ic, zscore, drop_extreme # Add other necessary imports from base_model
from master import MASTER # Ensure this import is correct

# Setup basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("[main_multi_index.py] Script started.") # DIAGNOSTIC PRINT

# Global configurations
FEATURE_START_COL = 3  # Assuming 'ticker', 'date', 'label' are the first three
LOOKBACK_WINDOW = 8
TRAIN_TEST_SPLIT_DATE = '2019-01-01' # Example split date

# Ensure Numba and other warnings are handled if necessary
warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=NumbaPerformanceWarning) # If using Numba and it's noisy

def get_relevant_weighted_log_returns(df, date_column='date', ticker_column='ticker', price_column='adj_close', volume_column='volume', lag=1, weight_ewm_span=20):
    # ... existing code ...
    pass

def create_sequences_multi_index(data, features_list, label_column, lookback_window):
    """
    Creates sequences for each ticker individually from a MultiIndex DataFrame.
    Ensures that sequences are only created where enough historical data exists for a ticker.
    """
    print(f"[main_multi_index.py] Starting create_sequences_multi_index for {len(data.index.get_level_values('ticker').unique())} tickers.") # DIAGNOSTIC PRINT
    all_X_list = []
    all_y_list = []
    new_multi_index_tuples = []

    grouped = data.groupby(level='ticker')
    processed_tickers = 0
    for ticker, group in grouped:
        # Ensure the group is sorted by date for sequence creation
        group = group.sort_index(level='date')
        
        df_features = group[features_list]
        df_labels = group[label_column]

        X_ticker, y_ticker = [], []
        ticker_index_tuples = []

        if len(df_features) >= lookback_window:
            for i in range(len(df_features) - lookback_window + 1):
                X_ticker.append(df_features.iloc[i:i + lookback_window].values)
                y_ticker.append(df_labels.iloc[i + lookback_window - 1]) # Label corresponds to the end of the window
                
                # Get the date and ticker for the *end* of the window
                current_date = group.index.get_level_values('date')[i + lookback_window - 1]
                ticker_index_tuples.append((ticker, current_date))

        if X_ticker: # Only add if sequences were created for this ticker
            all_X_list.extend(X_ticker)
            all_y_list.extend(y_ticker)
            new_multi_index_tuples.extend(ticker_index_tuples)
        processed_tickers += 1
        if processed_tickers % 100 == 0:
            print(f"[main_multi_index.py] Processed {processed_tickers}/{len(grouped)} tickers for sequencing.")


    if not all_X_list:
        print("[main_multi_index.py] No sequences created. Check data length and lookback window.") # DIAGNOSTIC PRINT
        return np.array([]), np.array([]), pd.MultiIndex.from_tuples([], names=['ticker', 'date'])

    final_X = np.array(all_X_list)
    final_y = np.array(all_y_list)
    final_index = pd.MultiIndex.from_tuples(new_multi_index_tuples, names=['ticker', 'date'])
    
    print(f"[main_multi_index.py] Finished create_sequences_multi_index. X shape: {final_X.shape}, y shape: {final_y.shape}, index length: {len(final_index)}") # DIAGNOSTIC PRINT
    return final_X, final_y, final_index

class DailyGroupedTimeSeriesDataset(Dataset):
    def __init__(self, X_sequences, y_targets, multi_index):
        """
        Dataset that groups data by unique dates from the multi_index.
        X_sequences: Numpy array of shape (num_sequences, lookback_window, num_features)
        y_targets: Numpy array of shape (num_sequences,)
        multi_index: Pandas MultiIndex with levels ('ticker', 'date'), aligned with X_sequences and y_targets.
                     The 'date' in the multi_index corresponds to the date of the label y_target.
        """
        print(f"[DailyGroupedTimeSeriesDataset] Initializing with X shape: {X_sequences.shape}, y shape: {y_targets.shape}, index length: {len(multi_index)}")
        self.X_sequences = X_sequences
        self.y_targets = y_targets
        self.multi_index = multi_index

        if not isinstance(multi_index, pd.MultiIndex):
            raise ValueError("multi_index must be a Pandas MultiIndex.")
        if not ('date' in multi_index.names and 'ticker' in multi_index.names):
            raise ValueError("multi_index must have 'ticker' and 'date' as level names.")

        self.unique_dates = sorted(self.multi_index.get_level_values('date').unique())
        self.data_by_date = {}

        # Pre-group data by date for faster __getitem__
        for date_val in self.unique_dates:
            date_mask = self.multi_index.get_level_values('date') == date_val
            self.data_by_date[date_val] = {
                'X': self.X_sequences[date_mask],
                'y': self.y_targets[date_mask]
            }
        print(f"[DailyGroupedTimeSeriesDataset] Initialized for {len(self.unique_dates)} unique dates.")

    def __len__(self):
        """Returns the number of unique days in the dataset."""
        return len(self.unique_dates)

    def __getitem__(self, idx):
        """
        Returns all features and labels for a single day.
        idx: An index representing a unique day.
        """
        selected_date = self.unique_dates[idx]
        day_data = self.data_by_date[selected_date]
        
        # Convert to tensors
        # X shape for a day: (num_stocks_on_this_day, lookback_window, num_features)
        # y shape for a day: (num_stocks_on_this_day,)
        features_tensor = torch.tensor(day_data['X'], dtype=torch.float32)
        labels_tensor = torch.tensor(day_data['y'], dtype=torch.float32)
        
        # print(f"[DailyGroupedTimeSeriesDataset] __getitem__({idx}) -> Date: {selected_date}, X_shape: {features_tensor.shape}, y_shape: {labels_tensor.shape}")
        return features_tensor, labels_tensor

    def get_index(self):
        """Returns the original multi_index, useful for aligning predictions."""
        return self.multi_index

def preprocess_data(df, features_list, label_column, lookback_window):
    """
    Main preprocessing function.
    Handles NaN imputation, feature scaling, and sequence creation.
    """
    print("[main_multi_index.py] Starting preprocess_data.") # DIAGNOSTIC PRINT
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
        df.index = pd.MultiIndex.from_tuples(
            [(ticker, pd.to_datetime(date_val)) for ticker, date_val in df.index],
            names=['ticker', 'date']
        )
        print("[main_multi_index.py] Converted date index to datetime.")


    # 1. Handle NaNs (grouped by ticker, then forward fill, then backward fill)
    print("[main_multi_index.py] Handling NaNs...")
    df[features_list] = df.groupby(level='ticker')[features_list].ffill().bfill()
    # df.dropna(subset=features_list + [label_column], inplace=True) # Drop rows if critical data still missing
    # A less aggressive approach: drop rows where label is NaN, or too many features are NaN
    initial_rows = len(df)
    df.dropna(subset=[label_column], inplace=True) # Label is critical
    # For features, allow some NaNs if not too many, or fill with 0/mean after ffill/bfill
    # For simplicity here, let's assume ffill/bfill handled most, or we accept some feature NaNs if model can handle
    # If any feature is ALL NaN for a ticker after ffill/bfill, those sequence windows will be problematic.
    # Let's fill any remaining NaNs in features with 0 after ffill/bfill, as a common strategy.
    df[features_list] = df[features_list].fillna(0)
    print(f"[main_multi_index.py] NaN handling complete. Rows before: {initial_rows}, after label drop & fill: {len(df)}")


    # 2. Feature Scaling (using StandardScaler, per feature, across all data for simplicity here)
    # More robust: fit scaler on training data only, then transform train/test.
    # For now, global scaling for demonstration.
    print("[main_multi_index.py] Scaling features...")
    scaler = StandardScaler()
    # Scale features. Avoid "SettingWithCopyWarning" by using .loc
    df.loc[:, features_list] = scaler.fit_transform(df[features_list])
    print("[main_multi_index.py] Feature scaling complete.")

    # 3. Create sequences
    X, y, seq_index = create_sequences_multi_index(df, features_list, label_column, lookback_window)

    if X.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No data after sequencing in preprocess_data. Exiting or handling.")
        # Depending on desired behavior, you might raise an error or return empty arrays
        return None, None, None 


    print(f"[main_multi_index.py] preprocess_data completed. X shape: {X.shape}, y shape: {y.shape}") # DIAGNOSTIC PRINT
    return X, y, seq_index, scaler # Return scaler if needed later for inverse transform or test set


def load_and_prepare_data(csv_path, feature_cols_start_idx, lookback, train_test_split_date_str):
    print(f"[main_multi_index.py] Starting load_and_prepare_data from: {csv_path}") # DIAGNOSTIC PRINT
    try:
        df = pd.read_csv(csv_path)
        print(f"[main_multi_index.py] CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"[main_multi_index.py] ERROR: CSV file not found at {csv_path}")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"[main_multi_index.py] ERROR: Could not read CSV: {e}")
        return None, None, None, None, None, None, None

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("[main_multi_index.py] ERROR: 'ticker' or 'date' column missing from CSV.")
        return None, None, None, None, None, None, None
        
    # Convert date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['ticker', 'date'], inplace=True)
    df.sort_index(inplace=True)
    print(f"[main_multi_index.py] Data indexed by ticker and date. Shape: {df.shape}")

    # Define features and label
    if 'label' not in df.columns:
        print("[main_multi_index.py] 'label' column not found. Attempting to generate based on future returns.")
        # Try 'closeadj' first, then 'adj_close', then 'close' as fallback for label generation
        label_source_col = None
        if 'closeadj' in df.columns:
            label_source_col = 'closeadj'
        elif 'adj_close' in df.columns: # Keep for backward compatibility or other datasets
            label_source_col = 'adj_close'
        elif 'close' in df.columns:
            label_source_col = 'close'
            print(f"[main_multi_index.py] WARNING: Using 'close' column to generate label as 'closeadj' or 'adj_close' not found.")
        
        if label_source_col:
            print(f"[main_multi_index.py] Generating 'label' as next day's pct_change of '{label_source_col}'.")
            df['label'] = df.groupby(level='ticker')[label_source_col].pct_change(1).shift(-1) # Example: next day's return
            df.dropna(subset=['label'], inplace=True) # Drop rows where label couldn't be computed (last day per ticker)
        else:
            print("[main_multi_index.py] ERROR: 'label' column missing and suitable price column ('closeadj', 'adj_close', 'close') not available to generate it.")
            return None, None, None, None, None, None, None
    
    label_column = 'label'
    all_cols = df.columns.tolist()
    feature_columns = [col for col in all_cols if col != label_column]

    if not feature_columns:
        print("[main_multi_index.py] ERROR: No feature columns identified.")
        return None, None, None, None, None, None, None
    print(f"[main_multi_index.py] Identified Label: '{label_column}'. Features: {feature_columns[:5]}... (Total: {len(feature_columns)})")


    # Preprocess: handles NaNs, scaling, and sequencing
    X, y, seq_idx, scaler = preprocess_data(df.copy(), feature_columns, label_column, lookback) # Use df.copy()
    
    if X is None or X.shape[0] == 0:
        print("[main_multi_index.py] ERROR: Preprocessing returned no data.")
        return None, None, None, None, None, None, None


    # Split data
    # Ensure seq_idx is sorted if it's not already from creation
    # seq_idx = seq_idx.sort_values() # Should be sorted by (ticker, date)
    
    # Get the date part of the multi-index for splitting
    dates_for_splitting = seq_idx.get_level_values('date')
    
    # Convert split date string to datetime
    split_datetime = pd.to_datetime(train_test_split_date_str)
    
    train_mask = dates_for_splitting < split_datetime
    test_mask = dates_for_splitting >= split_datetime

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_idx, test_idx = seq_idx[train_mask], seq_idx[test_mask]

    print(f"[main_multi_index.py] Data split. Train shape X: {X_train.shape}, y: {y_train.shape}. Test shape X: {X_test.shape}, y: {y_test.shape}") # DIAGNOSTIC PRINT
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("[main_multi_index.py] WARNING: Train or test set is empty after split. Check split date and data range.")
        # Decide if this is an error or acceptable (e.g., for a very short dataset)

    return X_train, y_train, train_idx, X_test, y_test, test_idx, scaler


def parse_args():
    print("[main_multi_index.py] Parsing arguments.") # DIAGNOSTIC PRINT
    parser = argparse.ArgumentParser(description="Train MASTER model on multi-index stock data.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--d_feat', type=int, default=None, help="Dimension of features (if None, inferred from data).") # d_feat is num_features
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden size of LSTM.")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--n_epochs_gru', type=int, default=10, help="Number of epochs for GRU base model.")
    parser.add_argument('--lr_gru', type=float, default=0.002, help="Learning rate for GRU base model.")
    parser.add_argument('--gpu', type=int, default=None, help="GPU ID to use (e.g., 0, 1). None for CPU.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--lookback', type=int, default=LOOKBACK_WINDOW, help="Lookback window for sequences.")
    parser.add_argument('--split_date', type=str, default=TRAIN_TEST_SPLIT_DATE, help="Date to split train/test data (YYYY-MM-DD).")
    parser.add_argument('--save_path', type=str, default='model_output/', help="Path to save trained models and results.")
    parser.add_argument('--model_type', type=str, default='GRU', choices=['GRU', 'LSTM', 'ALSTM'], help="Type of base model for SequenceModel.")


    args = parser.parse_args()
    print(f"[main_multi_index.py] Arguments parsed: {args}") # DIAGNOSTIC PRINT
    return args

def main():
    print("[main_multi_index.py] main() function started.") # DIAGNOSTIC PRINT
    args = parse_args()

    # Create save_path directory if it doesn't exist
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    print(f"[main_multi_index.py] Save path ensured: {args.save_path}")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Important for reproducibility
    print(f"[main_multi_index.py] Random seeds set. Seed: {args.seed}")


    # Load and prepare data
    print("[main_multi_index.py] Calling load_and_prepare_data...") # DIAGNOSTIC PRINT
    X_train, y_train, train_idx, X_test, y_test, test_idx, _ = load_and_prepare_data(
        args.csv, FEATURE_START_COL, args.lookback, args.split_date
    )

    if X_train is None or X_train.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No training data available after load_and_prepare_data. Exiting.") # DIAGNOSTIC PRINT
        return

    if X_test is None or X_test.shape[0] == 0:
        print("[main_multi_index.py] WARNING: No test data available. Proceeding with training only if this is intended.")
        # Depending on requirements, you might want to exit if test data is crucial.
        # For now, we'll allow it to proceed for training.

    # Determine d_feat (number of features) from the data
    if args.d_feat is None:
        d_feat = X_train.shape[2] # N, T, F -> F is the number of features
        print(f"[main_multi_index.py] Inferred d_feat (num_features): {d_feat}")
    else:
        d_feat = args.d_feat
        print(f"[main_multi_index.py] Using provided d_feat: {d_feat}")
        if d_feat != X_train.shape[2]:
            print(f"[main_multi_index.py] WARNING: Provided d_feat ({d_feat}) does not match data's feature dimension ({X_train.shape[2]}). Using data's dimension.")
            d_feat = X_train.shape[2]


    # Create Datasets and DataLoaders
    print("[main_multi_index.py] Creating train dataset...") # DIAGNOSTIC PRINT
    # Corrected: Pass the MultiIndex directly to the dataset
    train_dataset = DailyGroupedTimeSeriesDataset(X_train, y_train, train_idx)
    if X_test is not None and X_test.shape[0] > 0:
        print("[main_multi_index.py] Creating test dataset...") # DIAGNOSTIC PRINT
        test_dataset = DailyGroupedTimeSeriesDataset(X_test, y_test, test_idx)
    else:
        test_dataset = None
        print("[main_multi_index.py] No test dataset will be created.")


    # Initialize MASTER model
    print("[main_multi_index.py] Initializing MASTER model...") # DIAGNOSTIC PRINT
    # The MASTER class needs to be defined or imported correctly.
    # Assuming MASTER class takes these parameters. Adjust as per actual MASTER class definition.
    master_model = MASTER(
        d_feat=d_feat,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_epochs=args.n_epochs_gru, # This is for the base SequenceModel (GRU/LSTM)
        lr=args.lr_gru, # This is for the base SequenceModel
        GPU=args.gpu,
        seed=args.seed,
        model_type=args.model_type, # Pass model_type to MASTER
        save_path=args.save_path,
        save_prefix=f"master_model_{args.model_type}"
    )
    print("[main_multi_index.py] MASTER model initialized.") # DIAGNOSTIC PRINT


    # Train the model
    # The MASTER model's train method should internally use its SequenceModel and the datasets
    print("[main_multi_index.py] Starting MASTER model training...") # DIAGNOSTIC PRINT
    master_model.train_predict(
        train_data=train_dataset, # Pass the dataset object
        test_data=test_dataset    # Pass the dataset object, can be None
    )
    print("[main_multi_index.py] MASTER model training/prediction finished.") # DIAGNOSTIC PRINT

    # Example of how you might save predictions if train_predict returns them
    # For now, assume train_predict handles its own output/saving or prints metrics.

    print("[main_multi_index.py] main() function completed.") # DIAGNOSTIC PRINT

if __name__ == "__main__":
    print("[main_multi_index.py] Script execution started from __main__.") # DIAGNOSTIC PRINT
    main()
    print("[main_multi_index.py] Script execution finished from __main__.") # DIAGNOSTIC PRINT
