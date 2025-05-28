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
from torch.optim.lr_scheduler import ReduceLROnPlateau # For LR scheduling

from master import MASTERModel
from base_model import SequenceModel, DailyBatchSamplerRandom, calc_ic, zscore, drop_extreme # Add other necessary imports from base_model

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

# --- Helper class for Early Stopping (can be adapted from your previous base_model.py or kept simple) ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model_to_save):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_to_save):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        torch.save(model_to_save.state_dict(), self.path) # Save only the underlying nn.Module's state_dict
        self.val_loss_min = val_loss

def get_relevant_weighted_log_returns(df, date_column='date', ticker_column='ticker', price_column='adj_close', volume_column='volume', lag=1, weight_ewm_span=20):
    # ... existing code ...
    pass

def create_sequences_multi_index(data_multi_index, features_list, target_list, lookback_window, forecast_horizon, selected_tickers=None):
    """
    Creates sequences of features and corresponding targets for a multi-index DataFrame.

    Args:
        data_multi_index (pd.DataFrame): Multi-index DataFrame with 'ticker' and 'date' levels.
        features_list (list): List of column names to be used as features.
        target_list (list): List of column names to be used as targets.
        lookback_window (int): Number of time steps to look back for features.
        forecast_horizon (int): Number of time steps to forecast forward. 1 means predicting next step.
        selected_tickers (list, optional): A list of specific tickers to process. If None, all tickers are processed.

    Returns:
        tuple: (np.array of X sequences, np.array of y targets, list of (ticker, date) tuples for indices)
    """
    logger.debug(f"Starting create_sequences_multi_index for {len(data_multi_index.index.get_level_values('ticker').unique())} tickers.")
    all_X, all_y, all_indices = [], [], []
    
    tickers_to_process = data_multi_index.index.get_level_values('ticker').unique() if selected_tickers is None else selected_tickers
    
    processed_count = 0
    for ticker in tickers_to_process:
        if ticker in data_multi_index.index.get_level_values('ticker'):
            # Ensure data is sorted by date for each ticker
            df_ticker = data_multi_index.xs(ticker, level='ticker').sort_index()
            
            df_features = df_ticker[features_list]
            df_target = df_ticker[target_list]
            
            # Iterate to create sequences
            # The loop should go up to the point where the last feature window can be formed.
            # If len(df_features) is L and lookback_window is W,
            # the last valid start index i is such that i + W <= L, so i <= L - W.
            # range(L - W + 1) goes from 0 to L - W.
            for i in range(len(df_features) - lookback_window + 1):
                feature_end_idx = i + lookback_window
                # Target index is relative to the END of the feature window.
                # If forecast_horizon is 1, target is at feature_end_idx (i.e., next step after features).
                # target_idx in df_target corresponds to feature_end_idx -1 + forecast_horizon in df_features terms.
                target_date_actual_idx_in_df_target = feature_end_idx + forecast_horizon - 1

                # Ensure this target_date_actual_idx_in_df_target is within the bounds of df_target
                if target_date_actual_idx_in_df_target < len(df_target):
                    X_sequence = df_features.iloc[i:feature_end_idx].values
                    # Correctly select all target columns for the specific forecast point
                    y_sequence_targets = df_target.iloc[target_date_actual_idx_in_df_target].values 
                    
                    all_X.append(X_sequence)
                    all_y.append(y_sequence_targets) # y_target is now an array if target_list has multiple items
                    all_indices.append((ticker, df_target.index[target_date_actual_idx_in_df_target]))
                # else:
                    # logger.debug(f"Ticker {ticker}: Target index {target_date_actual_idx_in_df_target} out of bounds for df_target len {len(df_target)} at feature_end_idx {feature_end_idx}. Skipping sequence.")

        processed_count += 1
        if processed_count % 250 == 0:
            logger.debug(f"Processed {processed_count}/{len(tickers_to_process)} tickers for sequencing.")
            
    if not all_X:
        logger.warning("No sequences were created. Check data length, lookback window, and forecast horizon.")
        num_features = len(features_list)
        num_targets = len(target_list)
        return np.empty((0, lookback_window, num_features)), np.empty((0, num_targets)), []

    logger.debug(f"Finished create_sequences_multi_index. X shape: {np.array(all_X).shape}, y shape: {np.array(all_y).shape}, index length: {len(all_indices)}")
    return np.array(all_X), np.array(all_y), all_indices

class DailyGroupedTimeSeriesDataset(Dataset):
    def __init__(self, X_sequences, y_targets, multi_index, device=None, pin_memory=False, preprocess_labels=True):
        """
        Dataset that groups data by unique dates from the multi_index.
        Modified to keep tensors on CPU for multiprocessing compatibility.
        """
        print(f"[DailyGroupedTimeSeriesDataset] Initializing with X shape: {X_sequences.shape}, y shape: {y_targets.shape}, index length: {len(multi_index)}")
        print(f"[DailyGroupedTimeSeriesDataset] Optimization settings: device={device}, pin_memory={pin_memory}, preprocess_labels={preprocess_labels}")
        
        if not isinstance(multi_index, pd.MultiIndex):
            raise ValueError("multi_index must be a Pandas MultiIndex.")
        if not ('date' in multi_index.names and 'ticker' in multi_index.names):
            raise ValueError("multi_index must have 'ticker' and 'date' as level names.")

        self.multi_index = multi_index
        self.target_device = device  # Store target device but don't move tensors yet
        self.pin_memory = pin_memory
        self.preprocess_labels = preprocess_labels
        
        self.unique_dates = sorted(self.multi_index.get_level_values('date').unique())
        self.data_by_date = {}

        print(f"[DailyGroupedTimeSeriesDataset] Pre-converting data to tensors for {len(self.unique_dates)} unique dates...")
        
        # Pre-group data by date and convert to CPU tensors for multiprocessing compatibility
        for date_val in self.unique_dates:
            date_mask = self.multi_index.get_level_values('date') == date_val
            X_day = X_sequences[date_mask]
            y_day = y_targets[date_mask]
            
            # Convert to CPU tensors for multiprocessing compatibility
            X_tensor = torch.tensor(X_day, dtype=torch.float32)
            y_tensor = torch.tensor(y_day, dtype=torch.float32)
            
            # Pin memory if requested (faster CPU-GPU transfers)
            if self.pin_memory:
                X_tensor = X_tensor.pin_memory()
                y_tensor = y_tensor.pin_memory()
            
            day_data = {
                'X': X_tensor,
                'y_original': y_tensor
            }
            
            # Pre-process labels if requested
            if self.preprocess_labels and y_tensor.numel() > 0:
                # Import drop_extreme and zscore functions
                try:
                    from base_model import drop_extreme, zscore
                    
                    # Pre-compute label processing for training efficiency
                    if y_tensor.shape[0] >= 10:  # Same logic as training loop
                        with torch.no_grad():
                            y_tensor_for_processing = y_tensor.clone()
                            if self.target_device is None:  # Move to device temporarily for processing
                                y_tensor_for_processing = y_tensor_for_processing.cuda() if torch.cuda.is_available() else y_tensor_for_processing
                            
                            label_mask, y_dropped_extreme = drop_extreme(y_tensor_for_processing)
                            
                            # Fix shape issue: label_mask might be [N, 1] but we need [N] for indexing
                            if label_mask.dim() > 1:
                                label_mask = label_mask.squeeze(-1)
                            
                            y_processed = zscore(y_dropped_extreme)
                            
                            # Move back to original device/location
                            if self.target_device is None and torch.cuda.is_available():
                                label_mask = label_mask.cpu()
                                y_processed = y_processed.cpu()
                            
                            day_data['label_mask'] = label_mask
                            day_data['y_processed'] = y_processed
                    else:
                        # For small batches, use all samples (no drop_extreme)
                        label_mask = torch.ones(y_tensor.shape[0], dtype=torch.bool, device=y_tensor.device)
                        y_processed = zscore(y_tensor.clone())
                        day_data['label_mask'] = label_mask  
                        day_data['y_processed'] = y_processed
                        
                except ImportError:
                    print("[DailyGroupedTimeSeriesDataset] Warning: Could not import drop_extreme/zscore, skipping label preprocessing")
                    day_data['label_mask'] = None
                    day_data['y_processed'] = None
            else:
                day_data['label_mask'] = None
                day_data['y_processed'] = None
            
            self.data_by_date[date_val] = day_data
            
        print(f"[DailyGroupedTimeSeriesDataset] Tensor pre-conversion completed for {len(self.unique_dates)} unique dates.")

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
        
        # Data is already converted to tensors and potentially on GPU
        features_tensor = day_data['X']
        labels_tensor = day_data['y_original']
        
        # Return pre-processed labels if available
        processed_data = {
            'X': features_tensor,
            'y_original': labels_tensor,
            'label_mask': day_data.get('label_mask'),
            'y_processed': day_data.get('y_processed')
        }
        
        return processed_data

    def get_index(self):
        """Returns the original multi_index, useful for aligning predictions."""
        return self.multi_index

def preprocess_data(df, features_list, label_column, lookback_window, scaler=None, fit_scaler=False):
    """
    Main preprocessing function.
    Now expects features to already be cleaned and scaler to be pre-fitted.
    """
    print("[main_multi_index.py] Starting preprocess_data.") # DIAGNOSTIC PRINT
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
        df.index = pd.MultiIndex.from_tuples(
            [(ticker, pd.to_datetime(date_val)) for ticker, date_val in df.index],
            names=['ticker', 'date']
        )
        print("[main_multi_index.py] Converted date index to datetime.")

    # Features should already be cleaned by load_and_prepare_data
    print("[main_multi_index.py] Features assumed to be pre-cleaned (NaN/Inf handled).")

    # Handle NaNs in the label column (dropna) - this is crucial for target variables
    initial_rows_before_label_dropna = len(df)
    df.dropna(subset=[label_column], inplace=True) # This ensures labels are not NaN
    rows_dropped_for_label = initial_rows_before_label_dropna - len(df)
    if rows_dropped_for_label > 0:
        print(f"[main_multi_index.py] Dropped {rows_dropped_for_label} rows due to NaNs in label column '{label_column}'.")

    # Apply scaling using the pre-fitted scaler
    if scaler is not None and features_list and not df.empty:
        cols_to_scale = [col for col in features_list if col in df.columns]
        if cols_to_scale:
            df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale])
            print(f"[main_multi_index.py] Applied pre-fitted scaler to features: {len(cols_to_scale)} columns.")
        else:
            print("[main_multi_index.py] No features to scale with provided scaler.")
    elif scaler is None:
        print("[main_multi_index.py] No scaler provided - features not scaled.")
    
    print(f"[main_multi_index.py] Preprocessing complete. Rows remaining: {len(df)}")

    # Create sequences
    if not features_list:
        print("[main_multi_index.py] ERROR: features_list is empty before creating sequences in preprocess_data.")
        return None, None, None, scaler

    X, y, seq_index = create_sequences_multi_index(df, features_list, label_column, lookback_window)

    if X.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No data after sequencing in preprocess_data.")
        return None, None, None, scaler

    print(f"[main_multi_index.py] preprocess_data completed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, seq_index, scaler


def load_and_prepare_data(csv_path, feature_cols_start_idx, lookback, 
                          train_val_split_date_str, val_test_split_date_str,
                          gate_method='auto', gate_n_features=1, gate_percentage=0.1,
                          gate_start_index=None, gate_end_index=None):
    print(f"[main_multi_index.py] Starting load_and_prepare_data from: {csv_path}")
    print(f"[main_multi_index.py] Gate configuration: method={gate_method}, n_features={gate_n_features}, percentage={gate_percentage}, manual_indices=({gate_start_index}, {gate_end_index})")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"[main_multi_index.py] CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"[main_multi_index.py] ERROR: CSV file not found at {csv_path}")
        return (None,) * 12 # Corrected to 12
    except Exception as e:
        print(f"[main_multi_index.py] ERROR: Could not read CSV: {e}")
        return (None,) * 12 # Corrected to 12

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("[main_multi_index.py] ERROR: 'ticker' or 'date' column missing from CSV.")
        return (None,) * 12 # Corrected to 12
        
    try:
        df['date'] = pd.to_datetime(df['date']) 
    except ValueError:
        try:
            print("[main_multi_index.py] Initial pd.to_datetime failed, trying with infer_datetime_format=True")
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        except Exception as e_infer:
            print(f"[main_multi_index.py] ERROR: Could not parse 'date' column: {e_infer}")
            return (None,) * 12 # Corrected to 12
    except Exception as e_other:
            print(f"[main_multi_index.py] ERROR: pd.to_datetime failed with: {e_other}")
            return (None,) * 12 # Corrected to 12

    df.set_index(['ticker', 'date'], inplace=True)
    df.sort_index(inplace=True)
    print(f"[main_multi_index.py] Data indexed by ticker and date. Shape: {df.shape}")

    # --- Calculate and Add Market Volatility Feature ---
    market_feature_col_name = 'MRKT_AVG_CLOSEADJ_VOL20D'
    if 'closeadj' in df.columns:
        logger.info("Calculating market volatility feature based on 'closeadj'.")
        # Calculate daily average 'closeadj'
        daily_avg_closeadj = df.groupby(level='date')['closeadj'].mean()
        # Calculate 20-day rolling std of these daily averages
        market_volatility = daily_avg_closeadj.rolling(window=20, min_periods=1).std()
        # Only forward fill to avoid lookahead bias - NO bfill
        market_volatility = market_volatility.ffill()
        
        # Join this market feature back to the main df
        df = df.join(market_volatility.rename(market_feature_col_name), on='date')
        logger.info(f"Added '{market_feature_col_name}' to DataFrame. Example values: {df[market_feature_col_name].head()}")
        # Handle cases where some dates in df might not have had market_volatility (e.g., if df had dates outside daily_avg_closeadj range)
        # This should be minimal if daily_avg_closeadj covers all df dates.
        # The per-ticker ffill in preprocess_data will handle remaining NaNs for this new column.
        if df[market_feature_col_name].isnull().any():
            logger.warning(f"'{market_feature_col_name}' contains NaNs after join. These will be handled by per-ticker ffill later.")
    else:
        logger.error("'closeadj' column not found. Cannot calculate market volatility feature for the Gate. Exiting.")
        return (None,) * 12 # Corrected to 12
    # --- End of Market Feature Calculation ---

    if 'label' not in df.columns:
        label_source_col = None
        if 'closeadj' in df.columns: label_source_col = 'closeadj' # Primary source for label too
        elif 'adj_close' in df.columns: label_source_col = 'adj_close'
        elif 'close' in df.columns:
            label_source_col = 'close'
            print(f"[main_multi_index.py] WARNING: Using 'close' for label generation as 'closeadj' not found.")
        
        if label_source_col:
            print(f"[main_multi_index.py] Generating 'label' from '{label_source_col}'.")
            # FIXED: Generate forward-looking returns without shift(-1) to avoid lookahead bias
            # pct_change(1) at time t gives us the return from t-1 to t
            # The sequence creation logic will properly align this with features from t-lookback to t-1
            df['label'] = df.groupby(level='ticker')[label_source_col].pct_change(1)
            df.dropna(subset=['label'], inplace=True) 
            print(f"[main_multi_index.py] Shape after label generation: {df.shape}")
        else:
            print("[main_multi_index.py] ERROR: 'label' column missing and no suitable price column ('closeadj', 'adj_close', 'close') found.")
            return (None,) * 12 # Corrected to 12
    
    label_column = 'label'
    # Now, potential_feature_cols will include our new market_feature_col_name if it's numeric
    potential_feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    if label_column in potential_feature_cols:
        potential_feature_cols.remove(label_column)
    
    # Ensure our market feature is in the list if it was added and is numeric
    if market_feature_col_name not in potential_feature_cols and market_feature_col_name in df.columns and pd.api.types.is_numeric_dtype(df[market_feature_col_name]):
        # This case should ideally not happen if it was added correctly and is numeric.
        # But as a safeguard, add it if it was missed by select_dtypes for some reason.
        potential_feature_cols.append(market_feature_col_name)
    elif market_feature_col_name not in df.columns:
         logger.error(f"Market feature '{market_feature_col_name}' was unexpectedly not found in DataFrame columns before creating feature_columns list.")
         return (None,) * 12 # Corrected to 12 (and fixed syntax from (None,)*11)


    feature_columns = potential_feature_cols
    print(f"[main_multi_index.py] Initial feature columns identified (Total: {len(feature_columns)}). Includes market feature: {market_feature_col_name in feature_columns}")


    train_val_split_dt = pd.to_datetime(train_val_split_date_str)
    val_test_split_dt  = pd.to_datetime(val_test_split_date_str)
    PURGE_WINDOW       = lookback

    train_end_excl  = train_val_split_dt - pd.Timedelta(days=PURGE_WINDOW)
    valid_end_excl  = val_test_split_dt  - pd.Timedelta(days=PURGE_WINDOW)

    mask_dates = df.index.get_level_values('date')
    train_df  = df[mask_dates < train_end_excl].copy()
    valid_df  = df[(mask_dates >= train_val_split_dt) & (mask_dates <  valid_end_excl)].copy()
    test_df   = df[mask_dates >= val_test_split_dt].copy()

    def _drop_nan_cols(slice_df: pd.DataFrame, cols: list, threshold: float = 0.30):
        # Replace inf with nan first, then calculate nan percentage
        nan_share = slice_df[cols].replace([np.inf, -np.inf], np.nan).isna().sum() / len(slice_df)
        cols_to_drop = nan_share[nan_share > threshold].index.tolist()
        keep_cols    = [c for c in cols if c not in cols_to_drop]
        return keep_cols, cols_to_drop

    feature_columns, dropped_cols = _drop_nan_cols(train_df, feature_columns)

    if dropped_cols:
        logger.info(f"Dropping {len(dropped_cols)} cols (>30% NaNs/Infs) based on TRAIN slice: {dropped_cols}")
        train_df.drop(columns=dropped_cols, inplace=True, errors='ignore')
        valid_df.drop(columns=dropped_cols, inplace=True, errors='ignore')
        test_df.drop(columns=dropped_cols,  inplace=True, errors='ignore')

    if not feature_columns:
        logger.error("All numeric feature candidates were dropped – aborting.")
        return (None,) * 12 # Corrected to 12

    # --- Dynamic Gate Index Determination ---
    def determine_gate_indices(method, n_features_total, feature_cols, market_col_name):
        """Determine gate indices based on the specified method."""
        if method == 'manual':
            if gate_start_index is None or gate_end_index is None:
                logger.error("Manual gate method specified but gate_start_index or gate_end_index not provided.")
                return None, None
            if not (0 <= gate_start_index < n_features_total and gate_start_index < gate_end_index <= n_features_total):
                logger.error(f"Manual gate indices [{gate_start_index}:{gate_end_index}] are invalid for {n_features_total} features.")
                return None, None
            return gate_start_index, gate_end_index
            
        elif method == 'calculated_market':
            # Use the single calculated market feature
            try:
                gate_idx = feature_cols.index(market_col_name)
                return gate_idx, gate_idx + 1
            except ValueError:
                logger.error(f"Calculated market feature '{market_col_name}' not found in feature columns.")
                return None, None
                
        elif method == 'last_n':
            # Use the last N features as market features
            if gate_n_features > n_features_total:
                logger.error(f"Requested {gate_n_features} gate features but only {n_features_total} total features available.")
                return None, None
            start_idx = n_features_total - gate_n_features
            return start_idx, n_features_total
            
        elif method == 'percentage':
            # Use the last X% of features as market features
            n_market_features = max(1, int(n_features_total * gate_percentage))
            if n_market_features >= n_features_total:
                logger.warning(f"Percentage {gate_percentage} results in {n_market_features} market features, which is >= total features {n_features_total}. Using last feature only.")
                n_market_features = 1
            start_idx = n_features_total - n_market_features
            return start_idx, n_features_total
            
        elif method == 'auto':
            # Smart detection: try different strategies based on dataset size and feature names
            # Strategy 1: If we have the calculated market feature, use it
            if market_col_name in feature_cols:
                try:
                    gate_idx = feature_cols.index(market_col_name)
                    logger.info(f"Auto mode: Using calculated market feature '{market_col_name}' at index {gate_idx}")
                    return gate_idx, gate_idx + 1
                except ValueError:
                    pass
            
            # Strategy 2: Look for market-like feature names
            market_keywords = ['market', 'mrkt', 'index', 'vix', 'vol', 'volatility', 'macro', 'sentiment']
            for i, col_name in enumerate(feature_cols):
                if any(keyword.lower() in col_name.lower() for keyword in market_keywords):
                    logger.info(f"Auto mode: Found market-like feature '{col_name}' at index {i}, using as gate input")
                    return i, i + 1
            
            # Strategy 3: Based on dataset size, use different defaults
            if n_features_total >= 50:
                # Large dataset: use last 10% as market features (similar to paper's approach)
                n_market_features = max(1, int(n_features_total * 0.1))
                start_idx = n_features_total - n_market_features
                logger.info(f"Auto mode: Large dataset ({n_features_total} features), using last {n_market_features} features as market features")
                return start_idx, n_features_total
            else:
                # Small dataset: use last feature as market feature
                logger.info(f"Auto mode: Small dataset ({n_features_total} features), using last feature as market feature")
                return n_features_total - 1, n_features_total
        
        logger.error(f"Unknown gate method: {method}")
        return None, None
    
    gate_input_start_index, gate_input_end_index = determine_gate_indices(
        gate_method, len(feature_columns), feature_columns, market_feature_col_name
    )
    
    if gate_input_start_index is None or gate_input_end_index is None:
        logger.error("Failed to determine gate indices. Check gate configuration and dataset.")
        return (None,) * 12
    
    num_market_features_from_gate = gate_input_end_index - gate_input_start_index
    logger.info(f"Gate configuration successful: method='{gate_method}', indices=[{gate_input_start_index}:{gate_input_end_index}], market_features={num_market_features_from_gate}")
    logger.info(f"Market features for gate: {feature_columns[gate_input_start_index:gate_input_end_index]}")
    # --- End of Dynamic Gate Index Determination ---

    # First, handle NaN/Inf replacement and ffill for all splits before scaling
    def _preprocess_features_no_scaling(df_slice, feature_cols):
        """Handle NaN/Inf replacement and ffill without scaling"""
        if feature_cols and not df_slice.empty:
            # Step 1: Replace Infs with NaNs
            df_slice.loc[:, feature_cols] = df_slice[feature_cols].replace([np.inf, -np.inf], np.nan)
            # Step 2: Forward fill only (no backward fill)
            df_slice.loc[:, feature_cols] = df_slice.groupby(level='ticker', group_keys=False)[feature_cols].ffill()
            # Step 3: Fill remaining NaNs with 0 (after ffill)
            # This is a common strategy, but consider if per-feature median/mean imputation before ffill might be better for some cases.
            # For now, 0-fill is simple and matches potential previous states of the code.
            df_slice.loc[:, feature_cols] = df_slice[feature_cols].fillna(0)
        return df_slice

    # Apply NaN/Inf handling and ffill to all splits first
    train_df = _preprocess_features_no_scaling(train_df.copy(), feature_columns) # Use .copy() to avoid SettingWithCopyWarning
    valid_df = _preprocess_features_no_scaling(valid_df.copy(), feature_columns) if not valid_df.empty else valid_df
    test_df  = _preprocess_features_no_scaling(test_df.copy(), feature_columns) if not test_df.empty else test_df
    
    logger.info("Completed NaN/Inf replacement and ffill for train, valid, and test sets.")

    # --- Feature Scaling: RobustZScoreNorm + clip [-3, 3] --- 
    # Fit scaler ONLY on training data using median and MAD
    train_median = None
    train_mad = None
    scaler_fitted_successfully = False

    if feature_columns and not train_df.empty:
        cols_to_scale = [col for col in feature_columns if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col])]
        if cols_to_scale:
            logger.info(f"Calculating median and MAD for scaling on {len(cols_to_scale)} columns from training data.")
            train_median = train_df[cols_to_scale].median()
            # Calculate MAD: median of absolute deviations from the median. Add small epsilon to avoid division by zero.
            train_mad = (train_df[cols_to_scale] - train_median).abs().median().replace(0, 1e-8) # paper suggests .replace(0,1)
            
            # Apply scaling to training data
            train_df.loc[:, cols_to_scale] = ((train_df[cols_to_scale] - train_median) / train_mad).clip(-3, 3)
            logger.info("Applied RobustZScoreNorm + clip to training data.")
            scaler_fitted_successfully = True

            # Apply scaling to validation data using train_median and train_mad
            if not valid_df.empty and cols_to_scale:
                valid_cols_to_scale = [col for col in cols_to_scale if col in valid_df.columns]
                if valid_cols_to_scale:
                    valid_df.loc[:, valid_cols_to_scale] = ((valid_df[valid_cols_to_scale] - train_median[valid_cols_to_scale]) / train_mad[valid_cols_to_scale]).clip(-3, 3)
                    logger.info("Applied RobustZScoreNorm + clip to validation data using training set parameters.")
            
            # Apply scaling to test data using train_median and train_mad
            if not test_df.empty and cols_to_scale:
                test_cols_to_scale = [col for col in cols_to_scale if col in test_df.columns]
                if test_cols_to_scale:
                    test_df.loc[:, test_cols_to_scale] = ((test_df[test_cols_to_scale] - train_median[test_cols_to_scale]) / train_mad[test_cols_to_scale]).clip(-3, 3)
                    logger.info("Applied RobustZScoreNorm + clip to test data using training set parameters.")
        else:
            logger.warning("No numeric columns found in feature_columns for scaling in training data.")
    else:
        logger.warning("Training data is empty or no feature columns specified. Skipping scaling.")

    # The old scaler object is no longer used. We pass None for it.
    # The `preprocess_data` function, if it still expects a scaler, would need adjustment or to not apply scaling again.
    # For this change, we assume scaling is done here in load_and_prepare_data completely.
    # The X_train, y_train etc. are now generated by calling create_sequences_multi_index directly with the scaled dataframes.

    # --- Sequence Creation --- 
    # Create sequences from the preprocessed and scaled DataFrames
    # The `preprocess_data` function was previously called here. Now we directly call `create_sequences_multi_index`
    # as scaling is handled above. The `label_column` and `lookback` are needed.

    # Ensure label_column is defined (it should be from earlier in the function)
    if 'label_column' not in locals() and 'label_column' not in globals():
        # Attempt to redefine it if it got lost, though this indicates a flow issue.
        if 'label' in df.columns: label_column = 'label'
        else: 
            logger.error("label_column is not defined before sequence creation. Cannot proceed.")
            return (None,) * 12 # Adjusted for 12 return values if scaler was one of them.

    target_columns_for_sequence = [label_column] # create_sequences_multi_index expects a list of targets
    forecast_horizon = 1 # Assuming a forecast horizon of 1 for the label

    X_train, y_train, train_idx = create_sequences_multi_index(
        train_df, feature_columns, target_columns_for_sequence, lookback, forecast_horizon
    ) if not train_df.empty else (np.array([]), np.array([]), [])
    
    X_valid, y_valid, valid_idx = create_sequences_multi_index(
        valid_df, feature_columns, target_columns_for_sequence, lookback, forecast_horizon
    ) if not valid_df.empty else (np.array([]), np.array([]), [])
    
    X_test, y_test, test_idx = create_sequences_multi_index(
        test_df, feature_columns, target_columns_for_sequence, lookback, forecast_horizon
    ) if not test_df.empty else (np.array([]), np.array([]), [])

    logger.info(f"Data split summary after scaling and sequencing → Train:{X_train.shape[0] if X_train is not None else 0} "
                f"| Valid:{X_valid.shape[0] if X_valid is not None else 0} "
                f"| Test:{X_test.shape[0] if X_test is not None else 0}")

    # Return None for the scaler object as it's not a StandardScaler instance anymore and scaling is self-contained.
    # The function now returns 12 values, including the gate indices.
    return X_train, y_train, train_idx, \
           X_valid, y_valid, valid_idx, \
           X_test, y_test, test_idx, \
           None, gate_input_start_index, gate_input_end_index # Returning None for scaler object


def parse_args():
    print("[main_multi_index.py] Parsing arguments.")
    parser = argparse.ArgumentParser(description="Train MASTER model on multi-index stock data.")
    parser.add_argument('--csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--d_feat', type=int, default=None, help="Dimension of features (if None, inferred).")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for MASTERModel attention.")
    
    parser.add_argument('--d_model', type=int, default=256, help="Dimension of the model (Transformer).")
    parser.add_argument('--t_nhead', type=int, default=4, help="Heads for Temporal Attention.")
    parser.add_argument('--s_nhead', type=int, default=2, help="Heads for Cross-sectional Attention.")
    parser.add_argument('--beta', type=float, default=5.0, help="Beta for Gate mechanism or RegRankLoss.")
    
    # Gate configuration arguments
    parser.add_argument('--gate_method', type=str, default='auto', 
                       choices=['auto', 'calculated_market', 'last_n', 'percentage', 'manual'],
                       help="Method for determining gate indices: 'auto' (smart detection), 'calculated_market' (use single calculated market feature), 'last_n' (last N features), 'percentage' (last X%% of features), 'manual' (specify indices)")
    parser.add_argument('--gate_n_features', type=int, default=1, 
                       help="Number of features to use for gate when using 'last_n' method (default: 1)")
    parser.add_argument('--gate_percentage', type=float, default=0.1, 
                       help="Percentage of features to use for gate when using 'percentage' method (default: 0.1 = 10%%)")
    parser.add_argument('--gate_start_index', type=int, default=None,
                       help="Manual gate start index (0-based) when using 'manual' method")
    parser.add_argument('--gate_end_index', type=int, default=None,
                       help="Manual gate end index (exclusive, 0-based) when using 'manual' method")
    
    parser.add_argument('--gpu', type=int, default=None, help="GPU ID (e.g., 0). None for CPU.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--lookback', type=int, default=LOOKBACK_WINDOW, help="Lookback window.")
    parser.add_argument('--train_val_split_date', type=str, default=VALIDATION_SPLIT_DATE)
    parser.add_argument('--val_test_split_date', type=str, default=TRAIN_TEST_SPLIT_DATE)
    parser.add_argument('--save_path', type=str, default='model_output/')
    
    args = parser.parse_args()
    print(f"[main_multi_index.py] Arguments parsed: {args}")
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
    print("[main_multi_index.py] main() function started.")
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    import torch.multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError as e:
            logger.warning(f"Could not set multiprocessing method: {e}")
    
    args = parse_args()

    save_path = Path(args.save_path) 
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"paper_master_arch_best_model.pt" 
    print(f"[main_multi_index.py] Save path: {args.save_path}, Best model: {best_model_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"CUDA selected. Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.warning("!! CUDA NOT AVAILABLE OR NOT SELECTED. RUNNING ON CPU.        !!")
        logger.warning("!! THIS WILL BE VERY SLOW. Ensure PyTorch CUDA is installed, !!")
        logger.warning("!! a GPU is available, and use --gpu <ID> argument.         !!")
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    print(f"[main_multi_index.py] Seed: {args.seed}. Device: {device}")

    X_train, y_train, train_idx, \
    X_valid, y_valid, valid_idx, \
    X_test, y_test, test_idx, scaler_obj, \
    gate_input_start_index, gate_input_end_index = load_and_prepare_data(
        args.csv, FEATURE_START_COL, args.lookback, 
        args.train_val_split_date, args.val_test_split_date,
        args.gate_method, args.gate_n_features, args.gate_percentage,
        args.gate_start_index, args.gate_end_index
    )

    # Check if data loading and preparation was successful
    if X_train is None or gate_input_start_index is None: 
        logger.error("Data loading and preparation failed (e.g., feature mismatch for gate indices or other critical error). See logs above. Exiting.")
        return # Exit main function if critical data is missing

    d_feat_total = X_train.shape[2]
    if args.d_feat is not None: 
        if args.d_feat != d_feat_total:
            logger.warning(f"Provided d_feat ({args.d_feat}) from arguments != data's actual total feature dim ({d_feat_total}). Using data's dim.")
    
    logger.info(f"d_feat_total (num_features after processing, used by model): {d_feat_total}")
    logger.info(f"Gate indices passed to model: start={gate_input_start_index}, end={gate_input_end_index}")

    # Convert index lists to MultiIndex objects for compatibility with DailyGroupedTimeSeriesDataset
    def convert_to_multiindex(idx_list):
        """Convert list of (ticker, date) tuples to MultiIndex"""
        if idx_list and isinstance(idx_list, list) and len(idx_list) > 0:
            return pd.MultiIndex.from_tuples(idx_list, names=['ticker', 'date'])
        else:
            return pd.MultiIndex.from_tuples([], names=['ticker', 'date'])

    train_idx_multiindex = convert_to_multiindex(train_idx)
    valid_idx_multiindex = convert_to_multiindex(valid_idx) if valid_idx else pd.MultiIndex.from_tuples([], names=['ticker', 'date'])
    test_idx_multiindex = convert_to_multiindex(test_idx) if test_idx else pd.MultiIndex.from_tuples([], names=['ticker', 'date'])

    train_dataset = DailyGroupedTimeSeriesDataset(X_train, y_train, train_idx_multiindex, device, False, True)
    valid_dataset = None
    if X_valid is not None and X_valid.shape[0] > 0:
        valid_dataset = DailyGroupedTimeSeriesDataset(X_valid, y_valid, valid_idx_multiindex, device, False, True)
    else:
        logger.warning("No validation data. Early stopping and LR scheduling on validation loss will not be active.")
    
    test_dataset = None
    if X_test is not None and X_test.shape[0] > 0:
        test_dataset = DailyGroupedTimeSeriesDataset(X_test, y_test, test_idx_multiindex, device, False, True)

    logger.info(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid) if X_valid is not None else 0}, Test samples: {len(X_test) if X_test is not None else 0}")
    logger.info(f"Train unique days: {len(train_dataset)}, Valid unique days: {len(valid_dataset) if valid_dataset else 0}, Test unique days: {len(test_dataset) if test_dataset else 0}")

    model_wrapper = MASTERModel( 
        d_feat=d_feat_total, # Pass total features to the wrapper
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout, 
        S_dropout_rate=args.dropout,
        beta=args.beta, 
        gate_input_start_index=gate_input_start_index, 
        gate_input_end_index=gate_input_end_index,     
        n_epochs=args.epochs, 
        lr=args.lr,
        GPU=args.gpu, 
        seed=args.seed,
        save_path=str(save_path), 
        save_prefix=f"paper_master_arch_d{args.d_model}" 
    )
    
    pytorch_model = model_wrapper.model.to(device)
    optimizer = model_wrapper.optimizer 
    criterion = model_wrapper.loss_fn.to(device) 

    scheduler = None
    if valid_dataset:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    early_stopping = None
    if valid_dataset:
        early_stopping = EarlyStopping(patience=10, verbose=True, path=str(best_model_path), trace_func=logger.info)

    logger.info("Starting training loop with paper's label processing...")
    
    # Create DataLoaders with custom collate function for variable batch sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Increase from 8 to 16 for better GPU utilization
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=8,  # Match training batch size
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=True if device.type == 'cuda' else False,
            persistent_workers=False,
            collate_fn=custom_collate_fn,  # ADD THIS - was missing!
            drop_last=False
        )
    
    logger.info(f"Created DataLoaders with custom collate function, batch_size=16 for better GPU utilization")
    
    for epoch in range(args.epochs):
        epoch_train_loss = 0
        pytorch_model.train()
        processed_batches_train = 0
        
        # Use DataLoader for optimized iteration with batching
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # batch_data now contains padded tensors with attention masks
            X_batch = batch_data['X'].to(device, non_blocking=True)  # [batch_size, max_stocks, seq_len, features]
            y_batch_original = batch_data['y_original'].to(device, non_blocking=True)  # [batch_size, max_stocks, 1]
            attention_masks = batch_data['attention_mask'].to(device, non_blocking=True)  # [batch_size, max_stocks]
            
            # Move optional tensors to device as well
            label_mask_batch = batch_data.get('label_mask')
            y_processed_batch = batch_data.get('y_processed')
            if label_mask_batch is not None:
                label_mask_batch = label_mask_batch.to(device, non_blocking=True)
            if y_processed_batch is not None:
                y_processed_batch = y_processed_batch.to(device, non_blocking=True)
            
            # Squeeze y_batch_original if needed
            if y_batch_original.dim() > 2 and y_batch_original.shape[-1] == 1:
                y_batch_original = y_batch_original.squeeze(-1)  # [batch_size, max_stocks]
            
            # Initialize batch loss
            total_batch_loss = 0
            total_valid_samples = 0
            
            # Process all days in the batch simultaneously
            for day_idx in range(X_batch.shape[0]):
                X_day = X_batch[day_idx]  # [max_stocks, seq_len, features]
                y_day = y_batch_original[day_idx]  # [max_stocks]
                attention_mask = attention_masks[day_idx]  # [max_stocks]
                
                # Filter real stocks using attention mask
                X_day_real = X_day[attention_mask]  # [num_real_stocks, seq_len, features]
                y_day_real = y_day[attention_mask]  # [num_real_stocks]
                
                if y_day_real.numel() == 0:
                    continue
                
                # Handle pre-processed labels if available (now on same device)
                if label_mask_batch is not None and y_processed_batch is not None:
                    # Use pre-processed data, but filter by attention mask
                    label_mask_day = label_mask_batch[day_idx][attention_mask]  # No need for .to(device) now
                    y_processed_day = y_processed_batch[day_idx][attention_mask]
                    
                    if not torch.any(label_mask_day):
                        continue
                        
                    X_day_filtered = X_day_real[label_mask_day]
                    y_day_processed = y_processed_day[label_mask_day] if label_mask_day.dim() > 0 else y_processed_day
                else:
                    # Fallback to on-the-fly processing
                    num_stocks = y_day_real.shape[0]
                    if num_stocks >= 10:
                        # Use drop_extreme
                        label_mask, y_dropped = drop_extreme(y_day_real.clone())
                        if label_mask.dim() > 1:
                            label_mask = label_mask.squeeze(-1)
                        
                        if not torch.any(label_mask):
                            continue
                            
                        X_day_filtered = X_day_real[label_mask]
                        y_day_processed = zscore(y_dropped)
                    else:
                        # Use all stocks
                        X_day_filtered = X_day_real
                        y_day_processed = zscore(y_day_real.clone())
                
                if X_day_filtered.shape[0] == 0:
                    continue
                
                # Forward pass
                preds = pytorch_model(X_day_filtered)  # [num_filtered_stocks, 1]
                preds = preds.squeeze()  # [num_filtered_stocks]
                y_day_processed = y_day_processed.squeeze()  # [num_filtered_stocks]
                
                # Calculate loss for this day
                if preds.numel() > 0 and y_day_processed.numel() > 0:
                    day_loss = criterion(preds, y_day_processed)
                    if not (torch.isnan(day_loss) or torch.isinf(day_loss)):
                        total_batch_loss += day_loss
                        total_valid_samples += 1
            
            # Update weights if we have valid samples
            if total_valid_samples > 0:
                # Average loss across all valid samples in the batch
                avg_batch_loss = total_batch_loss / total_valid_samples
                avg_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(pytorch_model.parameters(), 3.0)
                optimizer.step()
                
                epoch_train_loss += avg_batch_loss.item()
                processed_batches_train += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Avg Loss: {avg_batch_loss.item():.6f}, Valid days: {total_valid_samples}")

        avg_epoch_train_loss = epoch_train_loss / processed_batches_train if processed_batches_train > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_epoch_train_loss:.6f}")

        if valid_loader:
            epoch_val_loss = 0
            pytorch_model.eval()
            processed_batches_valid = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(valid_loader):
                    X_batch_val = batch_data['X'].to(device, non_blocking=True)
                    y_batch_val_original = batch_data['y_original'].to(device, non_blocking=True)
                    attention_masks_val = batch_data['attention_mask'].to(device, non_blocking=True)
                    
                    # Move optional tensors to device as well
                    label_mask_batch_val = batch_data.get('label_mask')
                    y_processed_batch_val = batch_data.get('y_processed')
                    if label_mask_batch_val is not None:
                        label_mask_batch_val = label_mask_batch_val.to(device, non_blocking=True)
                    if y_processed_batch_val is not None:
                        y_processed_batch_val = y_processed_batch_val.to(device, non_blocking=True)
                    
                    # Squeeze y_batch if needed
                    if y_batch_val_original.dim() > 2 and y_batch_val_original.shape[-1] == 1:
                        y_batch_val_original = y_batch_val_original.squeeze(-1)
                    
                    total_val_loss = 0
                    total_val_samples = 0
                    
                    for day_idx in range(X_batch_val.shape[0]):
                        X_day_val = X_batch_val[day_idx]
                        y_day_val = y_batch_val_original[day_idx]
                        attention_mask_val = attention_masks_val[day_idx]
                        
                        # Filter real stocks
                        X_day_val_real = X_day_val[attention_mask_val]
                        y_day_val_real = y_day_val[attention_mask_val]
                        
                        if y_day_val_real.numel() == 0:
                            continue
                        
                        # Apply zscore (no drop_extreme for validation)
                        y_day_val_processed = zscore(y_day_val_real.clone())
                        
                        # Forward pass
                        preds_val = pytorch_model(X_day_val_real)
                        preds_val = preds_val.squeeze()
                        y_day_val_processed = y_day_val_processed.squeeze()
                        
                        if preds_val.numel() > 0 and y_day_val_processed.numel() > 0:
                            val_loss = criterion(preds_val, y_day_val_processed)
                            if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                                total_val_loss += val_loss
                                total_val_samples += 1
                    
                    if total_val_samples > 0:
                        avg_val_loss = total_val_loss / total_val_samples
                        epoch_val_loss += avg_val_loss.item()
                        processed_batches_valid += 1

            avg_epoch_val_loss = epoch_val_loss / processed_batches_valid if processed_batches_valid > 0 else 0
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {avg_epoch_val_loss:.6f}")

            if scheduler:
                scheduler.step(avg_epoch_val_loss)
            if early_stopping:  # early_stopping expects a loss (lower is better)
                if processed_batches_valid > 0:  # Only step if val loss was computed
                    early_stopping(avg_epoch_val_loss, pytorch_model) 
                    if early_stopping.early_stop:
                        logger.info("Early stopping triggered.")
                        break
                else:
                    logger.warning(f"Epoch {epoch+1}: No valid batches processed in validation, skipping early stopping/scheduler step.")
        else: 
            if (epoch + 1) % 10 == 0:
                 temp_save_path = save_path / f"model_epoch_{epoch+1}.pt"
                 torch.save(pytorch_model.state_dict(), str(temp_save_path))
                 logger.info(f"Saved model at epoch {epoch+1} to {temp_save_path}")

    if valid_dataset and early_stopping and Path(early_stopping.path).exists():
        logger.info(f"Loading best model from: {early_stopping.path}")
        pytorch_model.load_state_dict(torch.load(early_stopping.path, map_location=device))
    elif not valid_dataset: # If no validation, use the model from the last epoch of training
        logger.info("No validation set was used. Using model from the last training epoch for predictions.")
        # (pytorch_model is already in its last state)
    else:
        logger.info("No best model path found or validation not used. Using model from end of training for predictions.")


    # Generate predictions for backtesting using the (potentially best) trained model
    predictions_df = pd.DataFrame(columns=['date', 'ticker', 'prediction', 'actual_return'])
    if test_dataset:
        logger.info("Generating predictions on the test set...")
        
        # Create DataLoader for test set as well
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for CUDA compatibility
            pin_memory=False,  # Disable pin_memory when using pre-moved tensors
            persistent_workers=False  # Disable when num_workers=0
        )
        
        pytorch_model.eval()
        all_predictions_list = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                # Extract data from the batch - DataLoader returns dict with batch dimension
                X_day_test = batch_data['X'].squeeze(0)  # Remove batch dimension
                y_day_test_actuals = batch_data['y_original'].squeeze(0)
                
                current_date_for_day = test_dataset.unique_dates[batch_idx]
                day_specific_mask = test_dataset.multi_index.get_level_values('date') == current_date_for_day
                tickers_for_day = test_dataset.multi_index[day_specific_mask].get_level_values('ticker')

                # Move to device if not already there
                if X_day_test.device != device:
                    X_day_test = X_day_test.to(device, non_blocking=True)
                
                preds_day_test = pytorch_model(X_day_test).squeeze().cpu().numpy()
                actuals_day_test = y_day_test_actuals.cpu().numpy()

                if len(tickers_for_day) != len(preds_day_test):
                    logger.error(f"Data mismatch for date {current_date_for_day}: {len(tickers_for_day)} tickers vs {len(preds_day_test)} preds. Skipping.")
                    continue
                
                for ticker_val, pred_score, actual_label in zip(tickers_for_day, preds_day_test, actuals_day_test):
                    # Ensure scalar values for backtesting
                    pred_score_scalar = float(pred_score) if hasattr(pred_score, 'item') else float(pred_score)
                    actual_label_scalar = float(actual_label.item()) if hasattr(actual_label, 'item') else float(actual_label[0] if isinstance(actual_label, (list, np.ndarray)) and len(actual_label) > 0 else actual_label)
                    
                    all_predictions_list.append({
                        'date': current_date_for_day,
                        'ticker': ticker_val,
                        'prediction': pred_score_scalar,
                        'actual_return': actual_label_scalar 
                    })
        
        if all_predictions_list:
            predictions_df = pd.DataFrame(all_predictions_list)
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        else:
            logger.warning("No predictions were generated for the test set.")
    else:
        logger.info("No test dataset available, skipping prediction generation for backtesting.")


    # Perform backtesting
    if predictions_df is not None and not predictions_df.empty:
        logger.info(f"Received predictions DataFrame with shape: {predictions_df.shape}")
        perform_backtesting(
            predictions_df=predictions_df,
            N_values_list=[1, 2, 3, 5, 10],
            output_path=save_path,
            risk_free_rate=0.0 
        )
    elif test_dataset is None: # Redundant due to above check, but safe
        logger.info("No test dataset was available, skipping backtesting.")
    else: # predictions_df is empty but test_dataset existed
        logger.warning("Predictions DataFrame is empty after test set processing, skipping backtesting.")

    print("[main_multi_index.py] main() function completed.")

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable number of stocks per day.
    Pads all days in the batch to the maximum number of stocks and creates attention masks.
    """
    # Find the maximum number of stocks in this batch
    max_stocks = max([item['X'].shape[0] for item in batch])
    
    padded_X = []
    padded_y_original = []
    attention_masks = []
    label_masks = []
    y_processed_list = []
    
    for item in batch:
        X = item['X']  # Shape: [num_stocks, seq_len, features]
        y_original = item['y_original']  # Shape: [num_stocks, 1]
        
        num_stocks = X.shape[0]
        
        # Create attention mask (True for real stocks, False for padding)
        attention_mask = torch.zeros(max_stocks, dtype=torch.bool)
        attention_mask[:num_stocks] = True
        
        # Pad X and y to max_stocks
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            # Pad X: ensure padding tensor has same last 2 dimensions
            X_pad = torch.zeros(pad_size, X.shape[1], X.shape[2], dtype=X.dtype)
            X_padded = torch.cat([X, X_pad], dim=0)
            
            # Pad y_original: ensure padding tensor has same last dimension
            y_pad = torch.zeros(pad_size, y_original.shape[1], dtype=y_original.dtype)
            y_padded = torch.cat([y_original, y_pad], dim=0)
        else:
            X_padded = X
            y_padded = y_original
        
        padded_X.append(X_padded)
        padded_y_original.append(y_padded)
        attention_masks.append(attention_mask)
        
        # Handle pre-processed labels if available
        if 'label_mask' in item and item['label_mask'] is not None:
            label_mask = item['label_mask']
            y_processed = item['y_processed']
            
            # Pad label_mask and y_processed
            if num_stocks < max_stocks:
                pad_size = max_stocks - num_stocks
                
                # Pad label_mask (boolean tensor)
                label_mask_pad = torch.zeros(pad_size, dtype=torch.bool)
                label_mask_padded = torch.cat([label_mask, label_mask_pad], dim=0)
                
                # Pad y_processed: check if it's 1D or 2D and pad accordingly
                if y_processed.dim() == 1:
                    y_processed_pad = torch.zeros(pad_size, dtype=y_processed.dtype)
                    y_processed_padded = torch.cat([y_processed, y_processed_pad], dim=0)
                elif y_processed.dim() == 2:
                    y_processed_pad = torch.zeros(pad_size, y_processed.shape[1], dtype=y_processed.dtype)
                    y_processed_padded = torch.cat([y_processed, y_processed_pad], dim=0)
                else:
                    # Handle unexpected dimensions
                    y_processed_pad = torch.zeros(pad_size, dtype=y_processed.dtype)
                    y_processed_padded = torch.cat([y_processed.flatten()[:num_stocks], y_processed_pad], dim=0)
            else:
                label_mask_padded = label_mask
                y_processed_padded = y_processed
                
            label_masks.append(label_mask_padded)
            y_processed_list.append(y_processed_padded)
        else:
            # Create placeholder tensors with correct shape
            label_masks.append(torch.zeros(max_stocks, dtype=torch.bool))
            y_processed_list.append(torch.zeros(max_stocks, dtype=torch.float32))
    
    # Stack all tensors
    result = {
        'X': torch.stack(padded_X, dim=0),  # [batch_size, max_stocks, seq_len, features]
        'y_original': torch.stack(padded_y_original, dim=0),  # [batch_size, max_stocks, 1]
        'attention_mask': torch.stack(attention_masks, dim=0),  # [batch_size, max_stocks]
    }
    
    # Add processed labels - always include them now (even if some are placeholder)
    result['label_mask'] = torch.stack(label_masks, dim=0)  # [batch_size, max_stocks]
    result['y_processed'] = torch.stack(y_processed_list, dim=0)  # [batch_size, max_stocks]
    
    return result

if __name__ == "__main__":
    print("[main_multi_index.py] Script execution started from __main__.") # DIAGNOSTIC PRINT
    main()
    print("[main_multi_index.py] Script execution finished from __main__.") # DIAGNOSTIC PRINT
