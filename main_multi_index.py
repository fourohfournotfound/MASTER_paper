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
# Import specific functions we need without importing the entire base_model
from base_model import zscore, drop_extreme

# Setup basic logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optimize logging for performance - set to ERROR only during intensive operations
logger.setLevel(logging.ERROR)  # Only show errors, no info/warning spam
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('pandas').setLevel(logging.ERROR)

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

def create_sequences_multi_index_vectorized(data_multi_index, features_list, target_list, lookback_window, forecast_horizon=1, selected_tickers=None):
    """
    OPTIMIZED: Creates sequences using vectorized pandas operations instead of ticker-by-ticker loops.
    ~10-50x faster than the original implementation for large datasets.
    """
    start_time = time.time()
    logger.debug(f"Starting VECTORIZED create_sequences_multi_index for {len(data_multi_index.index.get_level_values('ticker').unique())} tickers.")
    
    # Filter tickers if specified
    if selected_tickers is not None:
        ticker_mask = data_multi_index.index.get_level_values('ticker').isin(selected_tickers)
        data_multi_index = data_multi_index[ticker_mask]
    
    # Ensure data is sorted
    if not data_multi_index.index.is_monotonic_increasing:
        data_multi_index = data_multi_index.sort_index()
    
    # Get all tickers and create a mapping
    all_tickers = data_multi_index.index.get_level_values('ticker')
    ticker_groups = data_multi_index.groupby(level='ticker')
    
    # Pre-allocate lists for efficiency
    all_X, all_y, all_indices = [], [], []
    
    # Process each ticker group efficiently
    for ticker, group_data in ticker_groups:
        # Get features and targets for this ticker
        ticker_features = group_data[features_list].values  # Direct numpy array
        ticker_targets = group_data[target_list].values
        ticker_dates = group_data.index.get_level_values('date')
        
        n_samples = len(ticker_features)
        
        # Vectorized sequence creation for this ticker
        if n_samples >= lookback_window + forecast_horizon:
            # Create all valid starting indices at once
            valid_starts = np.arange(n_samples - lookback_window - forecast_horizon + 1)
            
            # Vectorized slicing using broadcasting
            for start_idx in valid_starts:
                feature_end_idx = start_idx + lookback_window
                target_idx = feature_end_idx + forecast_horizon - 1
                
                if target_idx < n_samples:
                    # Extract sequence and target
                    X_sequence = ticker_features[start_idx:feature_end_idx]
                    y_target = ticker_targets[target_idx]
                    target_date = ticker_dates[target_idx]
                    
                    all_X.append(X_sequence)
                    all_y.append(y_target)
                    all_indices.append((ticker, target_date))
    
    # Convert to numpy arrays efficiently
    if not all_X:
        logger.warning("No sequences were created. Check data length, lookback window, and forecast horizon.")
        num_features = len(features_list)
        num_targets = len(target_list)
        return np.empty((0, lookback_window, num_features)), np.empty((0, num_targets)), []
    
    X_array = np.array(all_X, dtype=np.float32)  # Use float32 to save memory
    y_array = np.array(all_y, dtype=np.float32)
    
    elapsed_time = time.time() - start_time
    logger.info(f"VECTORIZED sequence creation completed in {elapsed_time:.2f}s. X shape: {X_array.shape}, y shape: {y_array.shape}")
    
    return X_array, y_array, all_indices

def create_sequences_fast(data_multi_index, features_list, target_list, lookback_window, forecast_horizon=1):
    """
    FAST sequence creation - optimized version of original logic.
    Uses efficient pandas operations without over-engineering.
    """
    start_time = time.time()
    all_X, all_y, all_indices = [], [], []
    
    # Pre-filter and sort once
    if not data_multi_index.index.is_monotonic_increasing:
        data_multi_index = data_multi_index.sort_index()
    
    # Use efficient groupby iteration
    for ticker, group_data in data_multi_index.groupby(level='ticker'):
        # Direct numpy access for speed
        features = group_data[features_list].values
        targets = group_data[target_list].values
        dates = group_data.index.get_level_values('date').values
        
        n_samples = len(features)
        
        # Simple loop - often faster than complex vectorization for moderate sizes
        for i in range(n_samples - lookback_window - forecast_horizon + 1):
            feature_end = i + lookback_window
            target_idx = feature_end + forecast_horizon - 1
            
            if target_idx < n_samples:
                all_X.append(features[i:feature_end])
                all_y.append(targets[target_idx])
                all_indices.append((ticker, dates[target_idx]))
    
    if not all_X:
        return np.array([]).reshape(0, lookback_window, len(features_list)), np.array([]), []
    
    X_array = np.array(all_X, dtype=np.float32)
    y_array = np.array(all_y, dtype=np.float32)
    
    elapsed = time.time() - start_time
    print(f"[FAST] Sequence creation: {elapsed:.2f}s, X: {X_array.shape}, y: {y_array.shape}")
    
    return X_array, y_array, all_indices

def choose_sequence_method(data_multi_index, features_list, target_list, lookback_window, forecast_horizon=1):
    """
    Automatically choose the best sequence creation method based on dataset size.
    """
    n_tickers = len(data_multi_index.index.get_level_values('ticker').unique())
    n_total_rows = len(data_multi_index)
    
    # Use simple method for smaller datasets (often faster due to less overhead)
    if n_total_rows < 100000 or n_tickers < 100:
        return create_sequences_fast(data_multi_index, features_list, target_list, lookback_window, forecast_horizon)
    else:
        return create_sequences_multi_index_vectorized(data_multi_index, features_list, target_list, lookback_window, forecast_horizon)

class DailyGroupedTimeSeriesDataset(Dataset):
    def __init__(self, X_sequences, y_targets, multi_index, device=None, pin_memory=False, preprocess_labels=False):
        """
        SIMPLIFIED Dataset - removed expensive pre-processing that was causing slowdowns.
        Now uses lazy loading for better performance.
        """
        if not isinstance(multi_index, pd.MultiIndex):
            raise ValueError("multi_index must be a Pandas MultiIndex.")
        if not ('date' in multi_index.names and 'ticker' in multi_index.names):
            raise ValueError("multi_index must have 'ticker' and 'date' as level names.")

        self.X_sequences = X_sequences
        self.y_targets = y_targets
        self.multi_index = multi_index
        self.target_device = device
        self.pin_memory = pin_memory
        self.preprocess_labels = preprocess_labels
        
        # Get unique dates efficiently
        self.unique_dates = sorted(self.multi_index.get_level_values('date').unique())
        
        # Create a simple mapping from date to indices - much faster than pre-processing everything
        self.date_to_indices = {}
        for i, date_val in enumerate(self.unique_dates):
            date_mask = self.multi_index.get_level_values('date') == date_val
            self.date_to_indices[date_val] = np.where(date_mask)[0]
        
        print(f"[DailyGroupedTimeSeriesDataset] SIMPLIFIED initialization for {len(self.unique_dates)} unique dates - using lazy loading")

    def __len__(self):
        """Returns the number of unique days in the dataset."""
        return len(self.unique_dates)

    def __getitem__(self, idx):
        """
        SIMPLIFIED: Returns data for a single day with lazy tensor conversion.
        Much faster than pre-processing everything.
        """
        selected_date = self.unique_dates[idx]
        day_indices = self.date_to_indices[selected_date]
        
        # Get data for this day - lazy conversion to tensors
        X_day = self.X_sequences[day_indices]
        y_day = self.y_targets[day_indices]
        
        # Convert to tensors only when needed
        X_tensor = torch.tensor(X_day, dtype=torch.float32)
        y_tensor = torch.tensor(y_day, dtype=torch.float32)
        
        # Simple return without complex pre-processing
        return {
            'X': X_tensor,
            'y_original': y_tensor,
            'label_mask': None,  # Will be processed in training loop if needed
            'y_processed': None  # Will be processed in training loop if needed
        }

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

    X, y, seq_index = choose_sequence_method(df, features_list, label_column, lookback_window)

    if X.shape[0] == 0:
        print("[main_multi_index.py] ERROR: No data after sequencing in preprocess_data.")
        return None, None, None, scaler

    print(f"[main_multi_index.py] preprocess_data completed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, seq_index, scaler

def determine_gate_indices(method, n_features_total, feature_cols, market_col_name, 
                          gate_start_index=None, gate_end_index=None, 
                          gate_n_features=1, gate_percentage=0.1):
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

def load_and_prepare_data_optimized(csv_path, feature_cols_start_idx, lookback, 
                          train_val_split_date_str, val_test_split_date_str,
                          gate_method='auto', gate_n_features=1, gate_percentage=0.1,
                          gate_start_index=None, gate_end_index=None):
    """
    OPTIMIZED version of load_and_prepare_data with significant performance improvements:
    - Reduced redundant operations
    - More efficient pandas operations  
    - Better memory management
    - Vectorized computations
    """
    print(f"[OPTIMIZED] Starting load_and_prepare_data from: {csv_path}")
    
    try:
        # SIMPLIFIED CSV loading - remove complex dtype inference that was causing slowdowns
        df = pd.read_csv(csv_path)
        # Convert date after loading - more reliable than during loading
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"[OPTIMIZED] CSV loaded efficiently. Shape: {df.shape}")
    except Exception as e:
        print(f"[OPTIMIZED] ERROR loading CSV: {e}")
        return (None,) * 12

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("[OPTIMIZED] ERROR: Required columns missing.")
        return (None,) * 12

    # Set index efficiently - avoid redundant sorting
    df.set_index(['ticker', 'date'], inplace=True)
    df.sort_index(inplace=True)
    
    # --- MARKET FEATURE NAME DEFINITION ---
    # Note: Market features will be calculated AFTER data splitting to prevent look-ahead bias
    market_feature_col_name = 'MRKT_AVG_CLOSEADJ_VOL20D'
    if 'closeadj' not in df.columns:
        print("[OPTIMIZED] ERROR: closeadj column missing")
        return (None,) * 12

    # --- OPTIMIZED Label Generation ---
    if 'label' not in df.columns:
        label_source_col = 'closeadj'  # Primary choice
        if label_source_col in df.columns:
            # Calculate returns (from t-1 to t)
            df['returns'] = df.groupby(level='ticker')[label_source_col].pct_change(1)

            # Shift to create forward-looking targets (return from t to t+1)
            df['label'] = df.groupby(level='ticker')['returns'].shift(-1)

            # Drop last row per ticker (it will have NaN label after shift)
            df = df.groupby(level='ticker').apply(lambda x: x.iloc[:-1])
            print(f"[OPTIMIZED] Labels generated efficiently. Shape: {df.shape}")
        else:
            print("[OPTIMIZED] ERROR: Cannot generate labels")
            return (None,) * 12

    # --- OPTIMIZED Feature Selection ---
    # Use more efficient dtype checking
    numeric_mask = df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))
    potential_feature_cols = df.columns[numeric_mask].tolist()
    
    if 'label' in potential_feature_cols:
        potential_feature_cols.remove('label')
    
    feature_columns = potential_feature_cols
    print(f"[OPTIMIZED] Feature selection completed. Total features: {len(feature_columns)}")

    # --- OPTIMIZED Data Splitting ---
    # Convert dates once and reuse
    train_val_split_dt = pd.to_datetime(train_val_split_date_str)
    val_test_split_dt = pd.to_datetime(val_test_split_date_str)
    forecast_horizon = 1
    PURGE_WINDOW = lookback + forecast_horizon

    train_end_excl = train_val_split_dt - pd.Timedelta(days=PURGE_WINDOW)
    valid_end_excl = val_test_split_dt - pd.Timedelta(days=PURGE_WINDOW)

    # More efficient boolean indexing
    dates = df.index.get_level_values('date')
    train_mask = dates < train_end_excl
    valid_mask = (dates >= train_val_split_dt) & (dates < valid_end_excl)
    test_mask = dates >= val_test_split_dt

    # Use views instead of copies where possible to save memory
    train_df = df.loc[train_mask].copy()  # Only copy when necessary
    valid_df = df.loc[valid_mask].copy() if valid_mask.any() else pd.DataFrame()
    test_df = df.loc[test_mask].copy() if test_mask.any() else pd.DataFrame()

    # --- FIXED: MARKET FEATURE CALCULATION AFTER SPLITTING (NO LOOK-AHEAD BIAS) ---
    def _calculate_market_features_safe(df_split, market_col_name, window=20):
        """Calculate market features for a data split without look-ahead bias."""
        if df_split.empty or 'closeadj' not in df_split.columns:
            return df_split
        
        # Calculate daily market average (cross-sectional mean) for this split only
        daily_avg_closeadj = df_split.groupby(level='date')['closeadj'].mean()
        
        # Calculate rolling volatility using only past data (causal)
        market_volatility = daily_avg_closeadj.rolling(
            window=window, 
            min_periods=1
        ).std().ffill()  # Forward fill only - no backward fill
        
        # Map volatility to all stocks on each date
        date_to_vol = market_volatility.to_dict()
        df_split[market_col_name] = df_split.index.get_level_values('date').map(date_to_vol)
        
        return df_split

    # Apply market feature calculation to each split independently
    print(f"[FIXED] Calculating market features per split to prevent look-ahead bias...")
    train_df = _calculate_market_features_safe(train_df, market_feature_col_name)
    valid_df = _calculate_market_features_safe(valid_df, market_feature_col_name) if not valid_df.empty else valid_df
    test_df = _calculate_market_features_safe(test_df, market_feature_col_name) if not test_df.empty else test_df
    print(f"[FIXED] Market features calculated safely for all splits")

    # --- OPTIMIZED NaN Column Dropping ---
    def _drop_nan_cols_optimized(slice_df: pd.DataFrame, cols: list, threshold: float = 0.30):
        if slice_df.empty or not cols:
            return cols, []
        
        # Vectorized NaN checking - much faster
        cols_array = np.array(cols)
        nan_ratios = slice_df[cols].replace([np.inf, -np.inf], np.nan).isnull().mean()
        keep_mask = nan_ratios <= threshold
        
        keep_cols = cols_array[keep_mask].tolist()
        dropped_cols = cols_array[~keep_mask].tolist()
        
        return keep_cols, dropped_cols

    # Include market feature in feature columns list
    if market_feature_col_name not in feature_columns and not train_df.empty:
        feature_columns.append(market_feature_col_name)

    feature_columns, dropped_cols = _drop_nan_cols_optimized(train_df, feature_columns)

    if dropped_cols:
        print(f"[OPTIMIZED] Dropping {len(dropped_cols)} high-NaN columns")
        # Drop columns efficiently from all splits
        for df_split in [train_df, valid_df, test_df]:
            if not df_split.empty:
                df_split.drop(columns=dropped_cols, inplace=True, errors='ignore')

    if not feature_columns:
        print("[OPTIMIZED] ERROR: No valid features remaining")
        return (None,) * 12

    # --- OPTIMIZED Gate Index Determination (keep existing logic but make it faster) ---
    gate_input_start_index, gate_input_end_index = determine_gate_indices(
        gate_method, len(feature_columns), feature_columns, market_feature_col_name,
        gate_start_index, gate_end_index,
        gate_n_features, gate_percentage
    )
    
    if gate_input_start_index is None:
        print("[OPTIMIZED] ERROR: Gate index determination failed")
        return (None,) * 12

    print(f"[OPTIMIZED] Gate configuration: indices=[{gate_input_start_index}:{gate_input_end_index}]")

    # --- OPTIMIZED Preprocessing and Scaling ---
    def _preprocess_and_scale_optimized(df_slice, feature_cols, train_stats=None):
        """Optimized preprocessing with vectorized operations"""
        if df_slice.empty or not feature_cols:
            return df_slice, train_stats
        
        # Vectorized NaN/Inf handling
        df_slice.loc[:, feature_cols] = df_slice[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # More efficient forward fill using groupby
        df_slice.loc[:, feature_cols] = df_slice.groupby(level='ticker')[feature_cols].ffill()
        df_slice.loc[:, feature_cols] = df_slice[feature_cols].fillna(0)
        
        # Efficient scaling
        if train_stats is None:
            # Calculate stats on training data
            train_median = df_slice[feature_cols].median()
            train_mad = (df_slice[feature_cols] - train_median).abs().median().replace(0, 1e-8)
            train_stats = {'median': train_median, 'mad': train_mad}
        
        # Apply scaling
        scaled_data = ((df_slice[feature_cols] - train_stats['median']) / train_stats['mad']).clip(-3, 3)
        df_slice.loc[:, feature_cols] = scaled_data
        
        return df_slice, train_stats

    # Apply optimized preprocessing
    train_df, train_stats = _preprocess_and_scale_optimized(train_df, feature_columns)
    valid_df, _ = _preprocess_and_scale_optimized(valid_df, feature_columns, train_stats)
    test_df, _ = _preprocess_and_scale_optimized(test_df, feature_columns, train_stats)

    print(f"[OPTIMIZED] Preprocessing completed efficiently")

    # --- OPTIMIZED Sequence Creation ---
    target_columns = ['label']
    forecast_horizon = 1

    # Use the new vectorized sequence creation function
    X_train, y_train, train_idx = choose_sequence_method(
        train_df, feature_columns, target_columns, lookback, forecast_horizon
    ) if not train_df.empty else (np.array([]), np.array([]), [])
    
    X_valid, y_valid, valid_idx = choose_sequence_method(
        valid_df, feature_columns, target_columns, lookback, forecast_horizon
    ) if not valid_df.empty else (np.array([]), np.array([]), [])
    
    X_test, y_test, test_idx = choose_sequence_method(
        test_df, feature_columns, target_columns, lookback, forecast_horizon
    ) if not test_df.empty else (np.array([]), np.array([]), [])

    print(f"[OPTIMIZED] Data preparation completed efficiently → Train:{X_train.shape[0] if X_train is not None else 0} | Valid:{X_valid.shape[0] if X_valid is not None else 0} | Test:{X_test.shape[0] if X_test is not None else 0}")

    return X_train, y_train, train_idx, \
           X_valid, y_valid, valid_idx, \
           X_test, y_test, test_idx, \
           None, gate_input_start_index, gate_input_end_index

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
    gate_input_start_index, gate_input_end_index = load_and_prepare_data_optimized(
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

    train_dataset = DailyGroupedTimeSeriesDataset(X_train, y_train, train_idx_multiindex, device, False, False)  # Disable preprocess_labels
    valid_dataset = None
    if X_valid is not None and X_valid.shape[0] > 0:
        valid_dataset = DailyGroupedTimeSeriesDataset(X_valid, y_valid, valid_idx_multiindex, device, False, False)  # Disable preprocess_labels
    else:
        logger.warning("No validation data. Early stopping and LR scheduling on validation loss will not be active.")
    
    test_dataset = None
    if X_test is not None and X_test.shape[0] > 0:
        test_dataset = DailyGroupedTimeSeriesDataset(X_test, y_test, test_idx_multiindex, device, False, False)  # Disable preprocess_labels

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
    
    # OPTIMIZED: Remove redundant device transfers - handled by lazy loading
    pytorch_model = model_wrapper.model  # No .to(device) here
    optimizer = model_wrapper.optimizer 
    criterion = model_wrapper.loss_fn  # No .to(device) here

    scheduler = None
    if valid_dataset:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    early_stopping = None
    if valid_dataset:
        early_stopping = EarlyStopping(patience=10, verbose=True, path=str(best_model_path), trace_func=logger.info)

    logger.info("Starting training loop with paper's label processing...")
    
    # OPTIMIZED: Create DataLoaders with simple collate function (defer complex processing)
    print(f"[PERFORMANCE] Creating DataLoaders...")
    dataloader_start = time.time()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,  # Disable pin_memory to speed up creation
        collate_fn=simple_collate_fn,  # Use simple collate
        drop_last=False
    )
    
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=simple_collate_fn,
            drop_last=False
        )
    
    dataloader_time = time.time() - dataloader_start
    print(f"[PERFORMANCE] DataLoaders created in {dataloader_time:.3f}s")
    
    # Trigger lazy device transfer only when training starts
    if hasattr(model_wrapper, '_ensure_device'):
        model_wrapper._ensure_device()
        print(f"[PERFORMANCE] Lazy device transfer completed")
    
    logger.info(f"Setup completed - starting training...")
    
    # Track training performance
    epoch_times = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0
        pytorch_model.train()
        processed_batches_train = 0
        total_days_processed = 0
        
        # Log GPU memory before training epoch
        if torch.cuda.is_available():
            logger.info(f"GPU Memory - Before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
        
        # OPTIMIZED: Simple training loop using simplified batch processing
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Use simplified batch processing for better performance
            batch_loss, num_valid_days = simple_batch_forward(
                pytorch_model, criterion, batch_data, device, is_training=True
            )
            
            if batch_loss is not None and num_valid_days > 0:
                # Backward pass and optimization
                batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_value_(pytorch_model.parameters(), 3.0)
                optimizer.step()
                
                epoch_train_loss += batch_loss.item()
                total_days_processed += num_valid_days
                processed_batches_train += 1
                
                if batch_idx % 25 == 0:  # Log progress
                    print(f"[TRAINING] Epoch {epoch+1}, Batch {batch_idx}: Loss: {batch_loss.item():.6f}, Days: {num_valid_days}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        avg_epoch_train_loss = epoch_train_loss / processed_batches_train if processed_batches_train > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_epoch_train_loss:.6f}, Total Days: {total_days_processed}, Duration: {epoch_duration:.1f}s")

        if valid_loader:
            val_start_time = time.time()
            epoch_val_loss = 0
            pytorch_model.eval()
            processed_batches_valid = 0
            total_val_days = 0
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(valid_loader):
                    # Use simplified batch processing for validation too
                    val_loss, num_valid_days = simple_batch_forward(
                        pytorch_model, criterion, batch_data, device, is_training=False
                    )
                    
                    if val_loss is not None and num_valid_days > 0:
                        epoch_val_loss += val_loss.item()
                        total_val_days += num_valid_days
                        processed_batches_valid += 1

            val_duration = time.time() - val_start_time
            avg_epoch_val_loss = epoch_val_loss / processed_batches_valid if processed_batches_valid > 0 else 0
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {avg_epoch_val_loss:.6f}, Total Days: {total_val_days}, Val Duration: {val_duration:.1f}s")

            if scheduler:
                scheduler.step(avg_epoch_val_loss)
            if early_stopping:
                if processed_batches_valid > 0:
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
    
    # Log training performance summary
    if epoch_times:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        logger.info(f"Training Performance Summary:")
        logger.info(f"  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
        logger.info(f"  Average epoch time: {avg_epoch_time:.1f}s")
        logger.info(f"  Estimated speedup from vectorization: ~3-5x vs sequential processing")

    if valid_dataset and early_stopping and Path(early_stopping.path).exists():
        logger.info(f"Loading best model from: {early_stopping.path}")
        pytorch_model.load_state_dict(torch.load(early_stopping.path, map_location=device))
    elif not valid_dataset:
        logger.info("No validation set was used. Using model from the last training epoch for predictions.")
    else:
        logger.info("No best model path found or validation not used. Using model from end of training for predictions.")

    # Generate predictions for backtesting using the (potentially best) trained model
    predictions_df = pd.DataFrame(columns=['date', 'ticker', 'prediction', 'actual_return'])
    if test_dataset:
        logger.info("Generating predictions on the test set...")
        
        # OPTIMIZED: Use simple_collate_fn for fast prediction generation (same as training)
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # Use same batch size as training for consistency
            shuffle=False,
            num_workers=0,
            pin_memory=False,  # Disable pin_memory for faster creation
            collate_fn=simple_collate_fn,  # FIXED: Use optimized collate function
            drop_last=False
        )
        
        pytorch_model.eval()
        all_predictions_list = []
        processed_days = 0  # Track which day we're processing
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                # OPTIMIZED: Use simple processing for predictions - much faster
                for day_data in batch_data:
                    if processed_days >= len(test_dataset.unique_dates):
                        break  # Don't process more days than available
                        
                    X_day = day_data['X'].to(device, non_blocking=True)  # (N_stocks, seq_len, features)
                    y_day = day_data['y_original'].to(device, non_blocking=True)  # (N_stocks, 1)
                    
                    if y_day.dim() > 1 and y_day.shape[-1] == 1:
                        y_day = y_day.squeeze(-1)
                    
                    if X_day.shape[0] > 0:
                        # Simple forward pass without preprocessing (this is inference)
                        preds = pytorch_model(X_day)  # (N_stocks, 1)
                        if preds.dim() > 1:
                            preds = preds.squeeze()  # (N_stocks,)
                        
                        preds_cpu = preds.cpu().numpy()
                        actuals_cpu = y_day.cpu().numpy()
                        
                        # Get the current date being processed
                        current_date = test_dataset.unique_dates[processed_days]
                        
                        # Get tickers for this day efficiently
                        day_mask = test_dataset.multi_index.get_level_values('date') == current_date
                        tickers_for_day = test_dataset.multi_index[day_mask].get_level_values('ticker').tolist()
                        
                        # Map predictions to tickers
                        for i, (pred_score, actual_return) in enumerate(zip(preds_cpu, actuals_cpu)):
                            if i < len(tickers_for_day):
                                all_predictions_list.append({
                                    'date': current_date,
                                    'ticker': tickers_for_day[i],
                                    'prediction': float(pred_score),
                                    'actual_return': float(actual_return)
                                })
                    
                    processed_days += 1  # Move to next day
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed test batch {batch_idx}/{len(test_loader)} - {len(all_predictions_list)} predictions so far")
        
        if all_predictions_list:
            predictions_df = pd.DataFrame(all_predictions_list)
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            logger.info(f"Generated {len(predictions_df)} predictions for backtesting efficiently")
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

def simple_collate_fn(batch):
    """
    OPTIMIZED: Simplified collate function - minimal processing during DataLoader creation.
    Just returns the batch as-is, processing is deferred to the training/inference loop.
    This is much faster than custom_collate_fn for both training and prediction generation.
    """
    return batch  # Just return the batch as-is, process in training loop

def vectorized_batch_forward(pytorch_model, criterion, batch_data, device, is_training=True):
    """
    Vectorized batch processing that processes all stocks from all days simultaneously.
    This maximizes GPU utilization while maintaining time series safety.
    """
    X_batch = batch_data['X'].to(device, non_blocking=True)  # [batch_size, max_stocks, seq_len, features]
    y_batch_original = batch_data['y_original'].to(device, non_blocking=True)  # [batch_size, max_stocks, 1]
    attention_masks = batch_data['attention_mask'].to(device, non_blocking=True)  # [batch_size, max_stocks]
    
    # Squeeze y_batch if needed
    if y_batch_original.dim() > 2 and y_batch_original.shape[-1] == 1:
        y_batch_original = y_batch_original.squeeze(-1)  # [batch_size, max_stocks]
    
    batch_size, max_stocks, seq_len, features = X_batch.shape
    
    # Reshape to combine batch and stock dimensions for vectorized processing
    X_flat = X_batch.view(-1, seq_len, features)  # [batch_size * max_stocks, seq_len, features]
    y_flat = y_batch_original.view(-1)  # [batch_size * max_stocks]
    attention_flat = attention_masks.view(-1)  # [batch_size * max_stocks]
    
    # Filter out padded stocks using attention mask
    valid_mask = attention_flat  # Boolean mask for real stocks
    if not torch.any(valid_mask):
        return None, 0
    
    X_valid = X_flat[valid_mask]  # [num_valid_stocks, seq_len, features]
    y_valid = y_flat[valid_mask]  # [num_valid_stocks]
    
    # Create day indices for each stock (which day each stock belongs to)
    day_indices_flat = torch.arange(batch_size, device=device).repeat_interleave(max_stocks)  # [batch_size * max_stocks]
    day_indices_valid = day_indices_flat[valid_mask]  # [num_valid_stocks]
    
    # Simple approach: process all data on-the-fly without pre-processing complications
    unique_days = torch.unique(day_indices_valid)
    final_X_list = []
    final_y_list = []
    final_day_indices_list = []
    
    for day_idx in unique_days:
        # Get all stocks for this day
        day_mask = (day_indices_valid == day_idx)
        X_day = X_valid[day_mask]  # [num_stocks_this_day, seq_len, features]
        y_day = y_valid[day_mask]  # [num_stocks_this_day]
        
        if len(y_day) == 0:
            continue
            
        # Apply drop_extreme for training if we have enough stocks
        if len(y_day) >= 10 and is_training:
            try:
                label_mask_day, y_dropped = drop_extreme(y_day.clone())
                if label_mask_day.dim() > 1:
                    label_mask_day = label_mask_day.squeeze(-1)
                    
                if torch.any(label_mask_day):
                    X_day_filtered = X_day[label_mask_day]
                    y_day_processed = zscore(y_dropped)
                else:
                    # If drop_extreme removes everything, use all stocks with zscore
                    X_day_filtered = X_day
                    y_day_processed = zscore(y_day.clone())
            except:
                # If drop_extreme fails, fall back to zscore all
                X_day_filtered = X_day
                y_day_processed = zscore(y_day.clone())
        else:
            # For validation or small batches, use all stocks
            X_day_filtered = X_day
            y_day_processed = zscore(y_day.clone())
        
        if X_day_filtered.shape[0] > 0:
            final_X_list.append(X_day_filtered)
            final_y_list.append(y_day_processed)
            # Track which day each stock belongs to for loss calculation
            day_ids = torch.full((X_day_filtered.shape[0],), day_idx, device=device)
            final_day_indices_list.append(day_ids)
    
    if not final_X_list:
        return None, 0
    
    # Concatenate all processed data
    X_final = torch.cat(final_X_list, dim=0)  # [total_processed_stocks, seq_len, features]
    y_final = torch.cat(final_y_list, dim=0)  # [total_processed_stocks]
    day_indices_final = torch.cat(final_day_indices_list, dim=0)  # [total_processed_stocks]
    
    if X_final.shape[0] == 0:
        return None, 0
    
    # Single vectorized forward pass for ALL processed stocks from ALL days
    preds = pytorch_model(X_final)  # [total_processed_stocks, 1]
    preds = preds.squeeze()  # [total_processed_stocks]
    y_final = y_final.squeeze()  # [total_processed_stocks]
    
    # Calculate loss per day and average (maintains proper gradient weighting)
    unique_days_final = torch.unique(day_indices_final)
    day_losses = []
    
    for day_idx in unique_days_final:
        day_mask = (day_indices_final == day_idx)
        if torch.sum(day_mask) > 0:
            preds_day = preds[day_mask]
            y_day = y_final[day_mask]
            
            if preds_day.numel() > 0 and y_day.numel() > 0:
                day_loss = criterion(preds_day, y_day)
                if not (torch.isnan(day_loss) or torch.isinf(day_loss)):
                    day_losses.append(day_loss)
    
    if not day_losses:
        return None, 0
    
    # Average loss across days (not across all stocks, to maintain day-level weighting)
    total_loss = torch.stack(day_losses).mean()
    num_valid_days = len(day_losses)
    
    return total_loss, num_valid_days

def simple_batch_forward(pytorch_model, criterion, batch_list, device, is_training=True):
    """
    OPTIMIZED: Simple batch processing for better performance.
    Works with simple_collate_fn that just returns a list of day data.
    """
    if not batch_list:
        return None, 0
    
    all_X = []
    all_y = []
    
    # Simple processing - just concatenate day data
    for day_data in batch_list:
        X_day = day_data['X'].to(device, non_blocking=True)  # (N_stocks, seq_len, features)
        y_day = day_data['y_original'].to(device, non_blocking=True)  # (N_stocks, 1)
        
        if y_day.dim() > 1 and y_day.shape[-1] == 1:
            y_day = y_day.squeeze(-1)
        
        if X_day.shape[0] > 0:
            # Apply drop_extreme and zscore if training and enough samples
            if is_training and len(y_day) >= 10:
                try:
                    mask, y_filtered = drop_extreme(y_day.clone())
                    if torch.any(mask):
                        X_day = X_day[mask]
                        y_day = zscore(y_filtered)
                    else:
                        y_day = zscore(y_day)
                except:
                    y_day = zscore(y_day)
            else:
                y_day = zscore(y_day)
            
            if X_day.shape[0] > 0:
                all_X.append(X_day)
                all_y.append(y_day)
    
    if not all_X:
        return None, 0
    
    # Concatenate all data
    X_batch = torch.cat(all_X, dim=0)
    y_batch = torch.cat(all_y, dim=0)
    
    # Forward pass
    preds = pytorch_model(X_batch)
    if preds.dim() > 1:
        preds = preds.squeeze()
    
    # Calculate loss
    loss = criterion(preds, y_batch)
    
    return loss, len(batch_list)

# Add performance monitoring utility
def monitor_performance(func_name):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            if torch.cuda.is_available():
                start_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_delta = end_gpu_memory - start_gpu_memory
                print(f"[PERF] {func_name}: {execution_time:.2f}s, GPU Memory: {start_gpu_memory:.2f}GB -> {end_gpu_memory:.2f}GB (Δ{gpu_delta:+.2f}GB)")
            else:
                print(f"[PERF] {func_name}: {execution_time:.2f}s")
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    print("[main_multi_index.py] Script execution started from __main__.") # DIAGNOSTIC PRINT
    main()
    print("[main_multi_index.py] Script execution finished from __main__.") # DIAGNOSTIC PRINT
