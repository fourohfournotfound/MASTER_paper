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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # Added for plotting

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
TRAIN_TEST_SPLIT_DATE = '2019-01-01' # Date for val/test split
VALIDATION_SPLIT_DATE = '2018-01-01' # Date for train/val split

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

def preprocess_data(df, features_list, label_column, lookback_window, scaler=None, fit_scaler=False):
    """
    Main preprocessing function.
    Handles NaN imputation, feature scaling, and sequence creation.
    
    Args:
        scaler: If provided, use this fitted scaler. If None and fit_scaler=True, fit a new one.
        fit_scaler: Whether to fit the scaler on this data (should only be True for training data)
    """
    print("[main_multi_index.py] Starting preprocess_data.") # DIAGNOSTIC PRINT
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
        df.index = pd.MultiIndex.from_tuples(
            [(ticker, pd.to_datetime(date_val)) for ticker, date_val in df.index],
            names=['ticker', 'date']
        )
        print("[main_multi_index.py] Converted date index to datetime.")

    # 1. Handle NaNs in features (columns to drop were already handled globally before this function is called on splits)
    print("[main_multi_index.py] Preprocessing data split: Handling NaNs/Infs in features (replace inf, ffill, fill with 0).")
    
    if features_list and not df.empty:
        # Ensure Infs are NaNs for the current df slice for all feature columns
        df.loc[:, features_list] = df[features_list].replace([np.inf, -np.inf], np.nan)

        # Grouped forward fill for NaNs in features
        # Using group_keys=False can sometimes be safer for apply-like operations, though for ffill it might not be strictly necessary.
        df.loc[:, features_list] = df.groupby(level='ticker', group_keys=False)[features_list].ffill()
        
        # Fill any remaining NaNs in features with 0 (e.g., if a ticker's series started with NaN)
        df.loc[:, features_list] = df[features_list].fillna(0)
        print(f"[main_multi_index.py] Feature NaN/Inf handling complete for this split. Features processed: {len(features_list)}")
    elif df.empty:
        print("[main_multi_index.py] DataFrame is empty, skipping NaN/Inf handling for features.")
    elif not features_list:
        print("[main_multi_index.py] features_list is empty, skipping NaN/Inf handling for features.")


    # Handle NaNs in the label column (dropna) - this is crucial
    initial_rows_before_label_dropna = len(df)
    df.dropna(subset=[label_column], inplace=True) # This ensures labels are not NaN
    rows_dropped_for_label = initial_rows_before_label_dropna - len(df)
    if rows_dropped_for_label > 0:
        print(f"[main_multi_index.py] Dropped {rows_dropped_for_label} rows due to NaNs in label column '{label_column}'.")
    
    print(f"[main_multi_index.py] NaN handling for this data split complete. Rows remaining: {len(df)}")

    # 2. Feature Scaling
    print("[main_multi_index.py] Scaling features...")
    if fit_scaler:
        if not features_list:
            print("[main_multi_index.py] No features to scale (features_list is empty). Scaler will not be fitted.")
            # scaler remains as passed (e.g. None) or previous state
        elif df[features_list].empty:
             print("[main_multi_index.py] Data for scaling is empty. Scaler will not be fitted.")
        else:
            scaler = StandardScaler()
            df.loc[:, features_list] = scaler.fit_transform(df[features_list])
            print("[main_multi_index.py] Fitted new scaler and transformed features.")
    elif scaler is not None:
        if not features_list:
            print("[main_multi_index.py] No features to scale (features_list is empty).")
        elif df[features_list].empty:
            print("[main_multi_index.py] Data for scaling is empty, skipping transformation.")
        else:
            # Ensure only existing columns in df[features_list] are transformed
            # This can happen if scaler was fit on more features than currently available in a split
            # However, the global pruning should make features_list consistent.
            cols_to_scale = [col for col in features_list if col in df.columns]
            if cols_to_scale:
                 df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale])
                 print(f"[main_multi_index.py] Transformed features using provided scaler for columns: {cols_to_scale}.")
            else:
                print("[main_multi_index.py] No common features to scale with provided scaler or features_list is out of sync.")
    else:
        print("[main_multi_index.py] No scaling applied.")
        # scaler remains None if it was passed as None and fit_scaler is False

    # 3. Create sequences
    if not features_list:
        print("[main_multi_index.py] ERROR: features_list is empty before creating sequences in preprocess_data.")
        return None, None, None, scaler

    X, y, seq_index = create_sequences_multi_index(df, features_list, label_column, lookback_window)

    if X.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No data after sequencing in preprocess_data.")
        return None, None, None, scaler # Return scaler here

    print(f"[main_multi_index.py] preprocess_data completed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, seq_index, scaler


def load_and_prepare_data(csv_path, feature_cols_start_idx, lookback, 
                          train_val_split_date_str, val_test_split_date_str):
    print(f"[main_multi_index.py] Starting load_and_prepare_data from: {csv_path}") # DIAGNOSTIC PRINT
    try:
        df = pd.read_csv(csv_path)
        print(f"[main_multi_index.py] CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"[main_multi_index.py] ERROR: CSV file not found at {csv_path}")
        return None, None, None, None, None, None, None, None, None, None # Added scaler
    except Exception as e:
        print(f"[main_multi_index.py] ERROR: Could not read CSV: {e}")
        return None, None, None, None, None, None, None, None, None, None # Added scaler

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("[main_multi_index.py] ERROR: 'ticker' or 'date' column missing from CSV.")
        return None, None, None, None, None, None, None, None, None, None # Added scaler
        
    # Convert date column to datetime objects
    try:
        # Attempt to infer format first, then specify if errors or for consistency
        df['date'] = pd.to_datetime(df['date']) 
    except ValueError:
        try:
            # Fallback if direct conversion fails, try inferring (slower)
            print("[main_multi_index.py] Initial pd.to_datetime failed, trying with infer_datetime_format=True")
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        except Exception as e_infer:
            print(f"[main_multi_index.py] ERROR: Could not parse 'date' column: {e_infer}")
            return None, None, None, None, None, None, None, None, None, None
    except Exception as e_other: # Catch any other pd.to_datetime errors
            print(f"[main_multi_index.py] ERROR: pd.to_datetime failed with: {e_other}")
            return None, None, None, None, None, None, None, None, None, None


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
            # Ensure label generation is done per ticker to avoid leakage across tickers at boundaries
            df['label'] = df.groupby(level='ticker')[label_source_col].pct_change(1).shift(-1)
            # It's important to drop NaNs from labels. This might remove the last day for each ticker.
            # This dropna should happen before feature processing that might depend on alignment.
            df.dropna(subset=['label'], inplace=True) 
            print(f"[main_multi_index.py] Dropped rows where new label is NaN. Shape after label generation: {df.shape}")
        else:
            print("[main_multi_index.py] ERROR: 'label' column missing and suitable price column ('closeadj', 'adj_close', 'close') not available to generate it.")
            return None, None, None, None, None, None, None, None, None, None
    
    label_column = 'label'
    
    # Identify potential feature columns: all columns that are numeric and not the label.
    # Exclude known non-feature string columns explicitly if necessary.
    potential_feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remove the label column from this list if it's present (it should be, as it's numeric)
    if label_column in potential_feature_cols:
        potential_feature_cols.remove(label_column)
    
    feature_columns = potential_feature_cols

    # If there are other known non-numeric columns that might have slipped through 
    # (e.g., if they were accidentally converted to object type but should be numeric, or vice-versa),
    # they should be handled (e.g. dropped or converted) before this step.
    # Example: df.drop(columns=['any_other_string_column_not_for_features'], inplace=True, errors='ignore')

    print(f"[main_multi_index.py] Initial feature columns identified: {feature_columns[:5] if len(feature_columns) > 5 else feature_columns}... (Total: {len(feature_columns)})")

    # === Global NaN/Inf Column Pruning Step ===
    if feature_columns: 
        logger.info("Starting global NaN/Inf column pruning...")
        # Replace Inf with NaN globally in feature columns for accurate NaN count for pruning
        df.loc[:, feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

        columns_to_drop_globally = []
        for col in feature_columns:
            if col in df.columns: # Should always be true at this point
                nan_percentage = df[col].isnull().sum() / len(df)
                if nan_percentage > 0.3:
                    columns_to_drop_globally.append(col)
                    logger.info(f"Globally identified column '{col}' for removal due to {nan_percentage*100:.2f}% NaN values (threshold 30%).")
            # This else case should ideally not be hit if feature_columns are derived from df.columns
            # else: 
            #     logger.warning(f"Column '{col}' from feature_columns list not found in DataFrame during global pruning. Skipping.")

        if columns_to_drop_globally:
            df.drop(columns=columns_to_drop_globally, inplace=True)
            # Update feature_columns list
            feature_columns = [col for col in feature_columns if col not in columns_to_drop_globally]
            logger.info(f"Globally dropped {len(columns_to_drop_globally)} columns: {columns_to_drop_globally}.")
            logger.info(f"Remaining features after pruning: {len(feature_columns)}")
            if not feature_columns: # If all feature columns were dropped
                 logger.error("All feature columns were dropped due to high NaN content. No features remaining.")
                 print("[main_multi_index.py] ERROR: No feature columns remaining after global NaN threshold drop. Cannot proceed.")
                 return None, None, None, None, None, None, None, None, None, None 
        else:
            logger.info("No columns exceeded the 30% NaN threshold globally for pruning.")
    else: # No initial feature columns identified
        print("[main_multi_index.py] No numeric feature columns were identified initially. Skipping pruning.")
        logger.warning("No numeric feature columns identified initially. Cannot proceed with feature processing.")
        # If feature_columns is empty from the start, the previous error check for it should catch this.
        # This path (else of `if feature_columns:`) might be redundant if the earlier check is robust.
        # For safety, if we reach here and feature_columns is empty, it's an error.
        if not feature_columns: # Double check, as the outer if implies it was empty
             print("[main_multi_index.py] ERROR: No feature columns identified. Cannot proceed.")
             return None, None, None, None, None, None, None, None, None, None


    if not feature_columns: # Final check after pruning
        print("[main_multi_index.py] ERROR: No numeric feature columns identified or remaining after filtering.")
        return None, None, None, None, None, None, None, None, None, None
    print(f"[main_multi_index.py] Identified Label: '{label_column}'. Features after pruning: {feature_columns[:5] if len(feature_columns) > 5 else feature_columns}... (Total: {len(feature_columns)})")


    # First, create sequences without scaling to properly split the data
    # df.copy() is important here as preprocess_data modifies df inplace (e.g. label dropna)
    # feature_columns is now the pruned list.
    X_raw, y_raw, seq_idx, _ = preprocess_data(df.copy(), feature_columns, label_column, lookback, 
                                                scaler=None, fit_scaler=False)
    
    if X_raw is None or X_raw.shape[0] == 0:
        print("[main_multi_index.py] ERROR: Preprocessing returned no data.")
        return None, None, None, None, None, None, None, None, None, None

    # Split data first
    dates_for_splitting = seq_idx.get_level_values('date')
    train_val_split_datetime = pd.to_datetime(train_val_split_date_str)
    val_test_split_datetime = pd.to_datetime(val_test_split_date_str)

    train_mask = dates_for_splitting < train_val_split_datetime
    valid_mask = (dates_for_splitting >= train_val_split_datetime) & \
                 (dates_for_splitting < val_test_split_datetime)
    test_mask = dates_for_splitting >= val_test_split_datetime

    # Now fit scaler only on training data
    train_df = df[df.index.get_level_values('date') < train_val_split_datetime].copy()
    valid_df = df[(df.index.get_level_values('date') >= train_val_split_datetime) & 
                  (df.index.get_level_values('date') < val_test_split_datetime)].copy()
    test_df = df[df.index.get_level_values('date') >= val_test_split_datetime].copy()

    # Process each split with appropriate scaler usage
    X_train, y_train, train_idx, scaler = preprocess_data(train_df, feature_columns, label_column, 
                                                          lookback, scaler=None, fit_scaler=True)
    
    X_valid, y_valid, valid_idx, _ = preprocess_data(valid_df, feature_columns, label_column, 
                                                     lookback, scaler=scaler, fit_scaler=False) if not valid_df.empty else (None, None, None, None)
    
    X_test, y_test, test_idx, _ = preprocess_data(test_df, feature_columns, label_column, 
                                                  lookback, scaler=scaler, fit_scaler=False) if not test_df.empty else (None, None, None, None)

    print(f"[main_multi_index.py] Data split. ") # DIAGNOSTIC PRINT
    print(f"  Train: X{X_train.shape}, y{y_train.shape}, Idx{len(train_idx) if train_idx is not None else 0}")
    print(f"  Valid: X{X_valid.shape}, y{y_valid.shape}, Idx{len(valid_idx) if valid_idx is not None else 0}")
    print(f"  Test:  X{X_test.shape}, y{y_test.shape}, Idx{len(test_idx) if test_idx is not None else 0}")
    
    if X_train.shape[0] == 0 :
        print("[main_multi_index.py] WARNING: Training set is empty after split. Check split dates and data range.")
    if X_valid.shape[0] == 0 :
        print("[main_multi_index.py] WARNING: Validation set is empty after split. No validation will be performed during training.")
    if X_test.shape[0] == 0 :
        print("[main_multi_index.py] WARNING: Test set is empty after split. No final test evaluation will be performed.")
        # Decide if this is an error or acceptable

    return X_train, y_train, train_idx, \
           X_valid, y_valid, valid_idx, \
           X_test, y_test, test_idx, scaler


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
    parser.add_argument('--train_val_split_date', type=str, default=VALIDATION_SPLIT_DATE, help="Date to split train into train/validation (YYYY-MM-DD).")
    parser.add_argument('--val_test_split_date', type=str, default=TRAIN_TEST_SPLIT_DATE, help="Date to split validation and test data (YYYY-MM-DD).")
    parser.add_argument('--save_path', type=str, default='model_output/', help="Path to save trained models and results.")
    parser.add_argument('--model_type', type=str, default='GRU', choices=['GRU', 'LSTM', 'ALSTM'], help="Type of base model for SequenceModel.")


    args = parser.parse_args()
    print(f"[main_multi_index.py] Arguments parsed: {args}") # DIAGNOSTIC PRINT
    return args

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.0, target_return=0.0, periods_per_year=252):
    """
    Calculates the Sortino Ratio for a series of daily returns.
    """
    if daily_returns.empty or daily_returns.std() == 0: # Avoid division by zero or NaN results
        return np.nan

    excess_returns = daily_returns - (risk_free_rate / periods_per_year) # Daily risk-free rate
    downside_returns = excess_returns[excess_returns < target_return].copy() # Use target_return for downside deviation
    
    if downside_returns.empty: # No returns below target
        return np.nan if np.mean(excess_returns) <= 0 else np.inf # Or handle as very high if mean positive

    downside_std = np.std(downside_returns)
    if downside_std == 0: # No variation in downside returns
        return np.nan if np.mean(excess_returns) <= 0 else np.inf


    mean_portfolio_return_annualized = np.mean(daily_returns) * periods_per_year
    downside_std_annualized = downside_std * np.sqrt(periods_per_year)
    
    # Standard Sortino: (Mean Portfolio Return - Risk Free Rate) / Downside Deviation
    # Using mean daily return for numerator before annualizing, consistent with downside_std being from daily
    sortino = (np.mean(daily_returns) * periods_per_year - risk_free_rate) / downside_std_annualized
    # Alternative: (Mean(daily_returns) - daily_target_return) / daily_downside_std, then annualize.
    # Let's use annualized mean return and annualized downside deviation.
    # Mean excess return for numerator:
    # mean_daily_excess_return = np.mean(excess_returns)
    # sortino_ratio = (mean_daily_excess_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    
    # A common way: (Annualized Portfolio Return - Annualized Target Return) / Annualized Downside Deviation
    # Here, target_return is daily. Annualized target_return for comparison:
    # annualized_target = (1 + target_return)**periods_per_year - 1 if target_return !=0 else 0.0 (approx for small daily)
    # Or simply use annualized mean return and risk free rate.
    
    # Numerator: Annualized mean return - Annualized risk-free rate
    # Denominator: Annualized standard deviation of returns falling below the target return
    
    annualized_mean_return = np.mean(daily_returns) * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)

    if annualized_downside_std == 0:
        return np.nan if (annualized_mean_return - risk_free_rate) <= 0 else np.inf

    sortino = (annualized_mean_return - risk_free_rate) / annualized_downside_std
    return sortino


def perform_backtesting(predictions_df, N_values_list, output_path, risk_free_rate=0.0):
    """
    Performs backtesting based on model predictions.
    - predictions_df: DataFrame with 'date', 'ticker', 'prediction', 'actual_return'.
    - N_values_list: List of integers for N (e.g., [1, 3, 5]) for top/bottom N stocks.
    - output_path: Path to save plots and results.
    - risk_free_rate: Annual risk-free rate for Sortino ratio.
    """
    logger.info("Starting backtesting...")
    if predictions_df.empty:
        logger.warning("Predictions DataFrame is empty. Skipping backtesting.")
        return

    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df.sort_values(by=['date', 'ticker'], inplace=True)

    # Benchmark: Equally weighted portfolio of all stocks available each day
    benchmark_daily_returns = predictions_df.groupby('date')['actual_return'].mean()
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()

    results_summary = []

    for N in N_values_list:
        logger.info(f"--- Backtesting for N = {N} ---")
        
        long_only_returns_list = []
        short_only_returns_list = []
        long_short_returns_list = [] # Optional: Long-Short strategy
        dates_for_plot = []

        for date, group in predictions_df.groupby('date'):
            dates_for_plot.append(date)
            group_sorted = group.sort_values(by='prediction', ascending=False)
            
            # Ensure N is not greater than available stocks
            num_available_stocks = len(group_sorted)
            current_N_long = min(N, num_available_stocks)
            current_N_short = min(N, num_available_stocks)


            # Long-only strategy
            if current_N_long > 0:
                long_portfolio = group_sorted.head(current_N_long)
                long_return = long_portfolio['actual_return'].mean()
            else:
                long_return = 0.0
            long_only_returns_list.append(long_return)

            # Short-only strategy
            if current_N_short > 0:
                # For shorting, we want stocks predicted to perform poorly (lowest scores)
                # If 'prediction' is higher-is-better, then .tail() gives lowest scores.
                short_portfolio = group_sorted.tail(current_N_short)
                # Return from shorting is -1 * stock_return
                short_return = -short_portfolio['actual_return'].mean() 
            else:
                short_return = 0.0
            short_only_returns_list.append(short_return)
            
            # Long-Short strategy (example: 50% long, 50% short, dollar neutral conceptual)
            if current_N_long > 0 and current_N_short > 0 :
                ls_return = 0.5 * long_return + 0.5 * short_return # If short_return is already -mean(actual)
                                                                # If short_return was mean(actual_of_shorted_stocks), then 0.5 * long_R - 0.5 * short_portfolio_actual_R
                                                                # With current short_return def: 0.5 * mean(top_N_actual) + 0.5 * (-mean(bottom_N_actual))
            elif current_N_long > 0:
                ls_return = long_return # Only long leg if cannot short
            elif current_N_short > 0:
                ls_return = short_return # Only short leg if cannot long (unlikely if N_short > 0)
            else:
                ls_return = 0.0
            long_short_returns_list.append(ls_return)


        long_only_daily_returns = pd.Series(long_only_returns_list, index=pd.to_datetime(dates_for_plot))
        short_only_daily_returns = pd.Series(short_only_returns_list, index=pd.to_datetime(dates_for_plot))
        long_short_daily_returns = pd.Series(long_short_returns_list, index=pd.to_datetime(dates_for_plot))


        # Align benchmark returns with strategy dates
        aligned_benchmark_daily_returns = benchmark_daily_returns.reindex(long_only_daily_returns.index).fillna(0)


        # Cumulative Returns
        long_only_cumulative = (1 + long_only_daily_returns).cumprod()
        short_only_cumulative = (1 + short_only_daily_returns).cumprod()
        long_short_cumulative = (1 + long_short_daily_returns).cumprod()
        benchmark_cumulative_aligned = (1 + aligned_benchmark_daily_returns).cumprod()


        # Metrics
        total_return_long = (long_only_cumulative.iloc[-1] - 1) * 100 if not long_only_cumulative.empty else 0
        total_return_short = (short_only_cumulative.iloc[-1] - 1) * 100 if not short_only_cumulative.empty else 0
        total_return_ls = (long_short_cumulative.iloc[-1] - 1) * 100 if not long_short_cumulative.empty else 0
        total_return_benchmark = (benchmark_cumulative_aligned.iloc[-1] - 1) * 100 if not benchmark_cumulative_aligned.empty else 0
        
        annualized_return_long = ((1 + long_only_daily_returns.mean())**252 - 1) * 100 if not long_only_daily_returns.empty else 0
        annualized_return_short = ((1 + short_only_daily_returns.mean())**252 - 1) * 100 if not short_only_daily_returns.empty else 0
        annualized_return_ls = ((1 + long_short_daily_returns.mean())**252 - 1) * 100 if not long_short_daily_returns.empty else 0
        annualized_return_benchmark = ((1 + aligned_benchmark_daily_returns.mean())**252 - 1) * 100 if not aligned_benchmark_daily_returns.empty else 0

        sortino_long = calculate_sortino_ratio(long_only_daily_returns, risk_free_rate)
        sortino_short = calculate_sortino_ratio(short_only_daily_returns, risk_free_rate)
        sortino_ls = calculate_sortino_ratio(long_short_daily_returns, risk_free_rate)
        sortino_benchmark = calculate_sortino_ratio(aligned_benchmark_daily_returns, risk_free_rate)


        logger.info(f"  Long Top {N}: Total Return: {total_return_long:.2f}%, Ann. Return: {annualized_return_long:.2f}%, Sortino: {sortino_long:.2f}")
        logger.info(f"  Short Bottom {N}: Total Return: {total_return_short:.2f}%, Ann. Return: {annualized_return_short:.2f}%, Sortino: {sortino_short:.2f}")
        logger.info(f"  Long-Short Top/Bottom {N}: Total Return: {total_return_ls:.2f}%, Ann. Return: {annualized_return_ls:.2f}%, Sortino: {sortino_ls:.2f}")
        logger.info(f"  Benchmark: Total Return: {total_return_benchmark:.2f}%, Ann. Return: {annualized_return_benchmark:.2f}%, Sortino: {sortino_benchmark:.2f}")

        results_summary.append({
            'N': N,
            'Strategy': 'Long Only',
            'Total Return (%)': total_return_long,
            'Annualized Return (%)': annualized_return_long,
            'Sortino Ratio': sortino_long
        })
        results_summary.append({
            'N': N,
            'Strategy': 'Short Only',
            'Total Return (%)': total_return_short,
            'Annualized Return (%)': annualized_return_short,
            'Sortino Ratio': sortino_short
        })
        results_summary.append({
            'N': N,
            'Strategy': 'Long-Short',
            'Total Return (%)': total_return_ls,
            'Annualized Return (%)': annualized_return_ls,
            'Sortino Ratio': sortino_ls
        })


        # Plotting
        plt.figure(figsize=(12, 7))
        if not long_only_cumulative.empty:
            plt.plot(long_only_cumulative.index, long_only_cumulative, label=f'Long Top {N} Stocks')
        if not short_only_cumulative.empty:
            plt.plot(short_only_cumulative.index, short_only_cumulative, label=f'Short Bottom {N} Stocks')
        if not long_short_cumulative.empty:
            plt.plot(long_short_cumulative.index, long_short_cumulative, label=f'Long-Short Top/Bottom {N} Stocks')
        if not benchmark_cumulative_aligned.empty:
            plt.plot(benchmark_cumulative_aligned.index, benchmark_cumulative_aligned, label='Benchmark (Equal Weight)', linestyle='--')
        
        plt.title(f'Backtest Cumulative Returns (N={N})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plot_filename = Path(output_path) / f'backtest_cumulative_returns_N{N}.png'
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved backtest plot to {plot_filename}")

    results_df = pd.DataFrame(results_summary)
    results_filename = Path(output_path) / 'backtest_summary_metrics.csv'
    results_df.to_csv(results_filename, index=False)
    logger.info(f"Saved backtest summary metrics to {results_filename}")
    logger.info("Backtesting finished.")


def main():
    print("[main_multi_index.py] main() function started.") # DIAGNOSTIC PRINT
    args = parse_args()

    # Create save_path directory if it doesn't exist
    save_path = Path(args.save_path) # Ensure save_path is a Path object
    save_path.mkdir(parents=True, exist_ok=True)
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
    X_train, y_train, train_idx, \
    X_valid, y_valid, valid_idx, \
    X_test, y_test, test_idx, scaler_obj = load_and_prepare_data( # Renamed _ to scaler_obj
        args.csv, FEATURE_START_COL, args.lookback, 
        args.train_val_split_date, args.val_test_split_date # Pass new args
    )

    if X_train is None or X_train.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No training data available after load_and_prepare_data. Exiting.") # DIAGNOSTIC PRINT
        return

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
    train_dataset = DailyGroupedTimeSeriesDataset(X_train, y_train, train_idx)
    
    valid_dataset = None
    if X_valid is not None and X_valid.shape[0] > 0 and y_valid is not None and y_valid.shape[0] > 0 and valid_idx is not None and len(valid_idx) > 0:
        print("[main_multi_index.py] Creating validation dataset...") # DIAGNOSTIC PRINT
        valid_dataset = DailyGroupedTimeSeriesDataset(X_valid, y_valid, valid_idx)
    else:
        print("[main_multi_index.py] Validation data is empty or incomplete. No validation dataset will be created for training.")
        logger.warning("Validation data is empty or incomplete. No validation dataset will be created for training.")


    test_dataset = None
    if X_test is not None and X_test.shape[0] > 0 and y_test is not None and y_test.shape[0] > 0 and test_idx is not None and len(test_idx) > 0:
        print("[main_multi_index.py] Creating test dataset...") # DIAGNOSTIC PRINT
        test_dataset = DailyGroupedTimeSeriesDataset(X_test, y_test, test_idx)
    else:
        print("[main_multi_index.py] Test data is empty or incomplete. No test dataset will be created.")


    # Initialize MASTER model
    print("[main_multi_index.py] Initializing MASTER model...") # DIAGNOSTIC PRINT
    master_model = MASTER(
        d_feat=d_feat,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_epochs=args.n_epochs_gru, 
        lr=args.lr_gru, 
        GPU=args.gpu,
        seed=args.seed,
        model_type=args.model_type, 
        save_path=str(save_path), # Pass save_path as string
        save_prefix=f"master_model_{args.model_type}"
    )
    print("[main_multi_index.py] MASTER model initialized.") # DIAGNOSTIC PRINT


    # Train the model and get predictions
    print("[main_multi_index.py] Starting MASTER model training/prediction...") # DIAGNOSTIC PRINT
    predictions_df = master_model.train_predict(
        train_data=train_dataset, 
        valid_data=valid_dataset, # Pass validation dataset here
        test_data=test_dataset    
    )
    print("[main_multi_index.py] MASTER model training/prediction finished.") # DIAGNOSTIC PRINT

    if predictions_df is not None and not predictions_df.empty:
        logger.info(f"Received predictions DataFrame with shape: {predictions_df.shape}")
        perform_backtesting(
            predictions_df=predictions_df,
            N_values_list=[1, 2, 3, 5, 10], # Configurable list of N values
            output_path=save_path, # Use the Path object
            risk_free_rate=0.0 # Assume 0% annual risk-free rate
        )
    elif test_dataset is None:
        logger.info("No test dataset was available, skipping backtesting.")
    else:
        logger.warning("No predictions returned from model or predictions_df is empty, skipping backtesting.")


    print("[main_multi_index.py] main() function completed.") # DIAGNOSTIC PRINT

if __name__ == "__main__":
    print("[main_multi_index.py] Script execution started from __main__.") # DIAGNOSTIC PRINT
    main()
    print("[main_multi_index.py] Script execution finished from __main__.") # DIAGNOSTIC PRINT
