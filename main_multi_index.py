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
    print(f"[main_multi_index.py] Starting load_and_prepare_data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"[main_multi_index.py] CSV loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"[main_multi_index.py] ERROR: CSV file not found at {csv_path}")
        return (None,) * 11 # Adjusted for new return values
    except Exception as e:
        print(f"[main_multi_index.py] ERROR: Could not read CSV: {e}")
        return (None,) * 11

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("[main_multi_index.py] ERROR: 'ticker' or 'date' column missing from CSV.")
        return (None,) * 11
        
    try:
        df['date'] = pd.to_datetime(df['date']) 
    except ValueError:
        try:
            print("[main_multi_index.py] Initial pd.to_datetime failed, trying with infer_datetime_format=True")
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        except Exception as e_infer:
            print(f"[main_multi_index.py] ERROR: Could not parse 'date' column: {e_infer}")
            return (None,) * 11
    except Exception as e_other:
            print(f"[main_multi_index.py] ERROR: pd.to_datetime failed with: {e_other}")
            return (None,) * 11

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
        # Fill NaNs that can occur at the beginning/end of the rolling calculation
        market_volatility = market_volatility.ffill().bfill() 
        
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
        return (None,) * 11
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
            df['label'] = df.groupby(level='ticker')[label_source_col].pct_change(1).shift(-1)
            df.dropna(subset=['label'], inplace=True) 
            print(f"[main_multi_index.py] Shape after label generation: {df.shape}")
        else:
            print("[main_multi_index.py] ERROR: 'label' column missing and no suitable price column ('closeadj', 'adj_close', 'close') found.")
            return (None,) * 11
    
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
         return (None,)*11


    feature_columns = potential_feature_cols
    print(f"[main_multi_index.py] Initial feature columns identified (Total: {len(feature_columns)}). Includes market feature: {market_feature_col_name in feature_columns}")


    train_val_split_dt = pd.to_datetime(train_val_split_date_str)
    val_test_split_dt  = pd.to_datetime(val_test_split_date_str)
    PURGE_WINDOW       = lookback - 1

    train_end_excl  = train_val_split_dt - pd.Timedelta(days=PURGE_WINDOW)
    valid_end_excl  = val_test_split_dt  - pd.Timedelta(days=PURGE_WINDOW)

    mask_dates = df.index.get_level_values('date')
    train_df  = df[mask_dates < train_end_excl].copy()
    valid_df  = df[(mask_dates >= train_val_split_dt) & (mask_dates <  valid_end_excl)].copy()
    test_df   = df[mask_dates >= val_test_split_dt].copy()

    def _drop_nan_cols(slice_df: pd.DataFrame, cols: list, threshold: float = 0.30):
        nan_share = slice_df[cols].replace([np.inf, -np.inf], np.nan).isna().sum() / len(slice_df)
        cols_to_drop = nan_share[nan_share > threshold].index.tolist()
        keep_cols    = [c for c in cols if c not in cols_to_drop]
        return keep_cols, cols_to_drop

    feature_columns, dropped_cols = _drop_nan_cols(train_df, feature_columns)

    if dropped_cols:
        logger.info(f"Dropping {len(dropped_cols)} cols (>30% NaNs) based on TRAIN slice: {dropped_cols}")
        train_df.drop(columns=dropped_cols, inplace=True, errors='ignore')
        valid_df.drop(columns=dropped_cols, inplace=True, errors='ignore')
        test_df.drop(columns=dropped_cols,  inplace=True, errors='ignore')

    if not feature_columns:
        logger.error("All numeric feature candidates were dropped – aborting.")
        return (None,) * 11

    # --- Automatic Gate Index Determination (using the single calculated market feature) ---
    gate_input_start_index = None
    gate_input_end_index = None

    try:
        # The market_feature_col_name should be in the finalized feature_columns list.
        # If it was dropped by _drop_nan_cols, this will raise ValueError.
        gate_idx = feature_columns.index(market_feature_col_name)
        gate_input_start_index = gate_idx
        gate_input_end_index = gate_idx + 1 # Slice of length 1
        logger.info(f"Successfully set Gate to use internally calculated market feature: '{market_feature_col_name}'. "
                    f"Index in final feature list: {gate_idx}. Gate slice: [{gate_input_start_index}:{gate_input_end_index}]")
    except ValueError:
        logger.error(f"The internally calculated market feature '{market_feature_col_name}' was NOT FOUND in the final list of feature columns "
                       f"(it might have been dropped due to excessive NaNs on train split). Cannot configure Gate. Exiting.")
        return (None,) * 11
    # --- End of Gate Index Detection ---

    X_train, y_train, train_idx, scaler = preprocess_data(
        train_df, feature_columns, label_column, lookback, scaler=None, fit_scaler=True
    )
    X_valid, y_valid, valid_idx, _ = preprocess_data(
        valid_df, feature_columns, label_column, lookback, scaler=scaler, fit_scaler=False
    ) if not valid_df.empty else (None, None, None, None)
    X_test, y_test, test_idx, _ = preprocess_data(
        test_df, feature_columns, label_column, lookback, scaler=scaler, fit_scaler=False
    ) if not test_df.empty else (None, None, None, None)

    logger.info(f"Data split summary → Train:{0 if X_train is None else X_train.shape[0]} "
                f"| Valid:{0 if X_valid is None else X_valid.shape[0]} "
                f"| Test:{0 if X_test  is None else X_test.shape[0]}")

    return X_train, y_train, train_idx, \
           X_valid, y_valid, valid_idx, \
           X_test, y_test, test_idx, scaler, \
           gate_input_start_index, gate_input_end_index


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
    args = parse_args()

    save_path = Path(args.save_path) 
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"paper_master_arch_best_model.pt" 
    print(f"[main_multi_index.py] Save path: {args.save_path}, Best model: {best_model_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[main_multi_index.py] Seed: {args.seed}. Device: {device}")

    X_train, y_train, train_idx, \
    X_valid, y_valid, valid_idx, \
    X_test, y_test, test_idx, scaler_obj, \
    gate_input_start_index, gate_input_end_index = load_and_prepare_data(
        args.csv, FEATURE_START_COL, args.lookback, 
        args.train_val_split_date, args.val_test_split_date
    )

    if X_train is None or X_train.shape[0] == 0: 
        print("[main_multi_index.py] ERROR: No training data or failed market feature/gate index setup. Exiting.")
        return

    d_feat_total = X_train.shape[2]
    if args.d_feat is not None: # d_feat from args is expected to be d_feat_total
        if args.d_feat != d_feat_total:
            logger.warning(f"Provided d_feat ({args.d_feat}) != data's actual total feature dim ({d_feat_total}). Using data's dim.")
        # d_feat_total = args.d_feat # No, always use from data.
    
    print(f"[main_multi_index.py] d_feat_total (num_features including market): {d_feat_total}")
    
    if not (0 <= gate_input_start_index < d_feat_total and 
            gate_input_start_index < gate_input_end_index <= d_feat_total and 
            gate_input_end_index == gate_input_start_index + 1): # For single market feature
        logger.error(f"Internally determined Gate indices [{gate_input_start_index}, {gate_input_end_index}) "
                       f"are invalid for d_feat_total={d_feat_total}. Check market feature processing.")
        sys.exit(1)

    train_dataset = DailyGroupedTimeSeriesDataset(X_train, y_train, train_idx)
    valid_dataset = None
    if X_valid is not None and X_valid.shape[0] > 0:
        valid_dataset = DailyGroupedTimeSeriesDataset(X_valid, y_valid, valid_idx)
    else:
        logger.warning("No validation data. Early stopping and LR scheduling on validation loss will not be active.")
    
    test_dataset = None
    if X_test is not None and X_test.shape[0] > 0:
        test_dataset = DailyGroupedTimeSeriesDataset(X_test, y_test, test_idx)

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
    for epoch in range(args.epochs):
        epoch_train_loss = 0
        pytorch_model.train()
        processed_batches_train = 0
        for i in range(len(train_dataset)): 
            X_day, y_day_original = train_dataset[i]
            X_day, y_day_original = X_day.to(device), y_day_original.to(device)

            if y_day_original.numel() == 0 : # Skip if no labels for the day
                continue

            # Apply label processing as per paper for training
            # drop_extreme and zscore expect torch tensors
            label_mask, y_day_dropped_extreme = drop_extreme(y_day_original.clone()) 
            
            if not torch.any(label_mask): # if all labels were dropped
                # logger.debug(f"Epoch {epoch+1}, Day {i}: All labels dropped by drop_extreme. Skipping batch.")
                continue
            
            y_day_processed_for_loss = zscore(y_day_dropped_extreme) # CSZscoreNorm
            
            # Filter X_day based on the mask from drop_extreme
            X_day_filtered = X_day[label_mask]

            if X_day_filtered.shape[0] == 0: # If all corresponding features are gone
                # logger.debug(f"Epoch {epoch+1}, Day {i}: All features dropped after label mask. Skipping batch.")
                continue
                
            optimizer.zero_grad()
            preds_day_all = pytorch_model(X_day) # Get predictions for ALL X_day items
            preds_day_filtered_for_loss = preds_day_all[label_mask] # Filter predictions

            if preds_day_filtered_for_loss.shape[0] != y_day_processed_for_loss.shape[0]:
                logger.error(f"Epoch {epoch+1}, Day {i}: Mismatch after filtering. Preds: {preds_day_filtered_for_loss.shape}, Labels: {y_day_processed_for_loss.shape}. Skipping.")
                continue
            if preds_day_filtered_for_loss.numel() == 0: # If no elements left to calculate loss
                continue


            loss = criterion(preds_day_filtered_for_loss.squeeze(), y_day_processed_for_loss.squeeze()) 
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Epoch {epoch+1}, Day {i}: NaN or Inf loss detected. Skipping update. Preds: {preds_day_filtered_for_loss}, Labels: {y_day_processed_for_loss}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_value_(pytorch_model.parameters(), 3.0) # As per paper's base_model
            optimizer.step()
            epoch_train_loss += loss.item()
            processed_batches_train += 1
        
        avg_epoch_train_loss = epoch_train_loss / processed_batches_train if processed_batches_train > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_epoch_train_loss:.6f}")

        if valid_dataset:
            epoch_val_loss = 0
            pytorch_model.eval()
            processed_batches_valid = 0
            with torch.no_grad():
                for i in range(len(valid_dataset)):
                    X_day_val, y_day_val_original = valid_dataset[i]
                    X_day_val, y_day_val_original = X_day_val.to(device), y_day_val_original.to(device)

                    if y_day_val_original.numel() == 0:
                        continue
                    
                    # Apply CSZscoreNorm for validation labels (NO drop_extreme)
                    y_day_val_processed = zscore(y_day_val_original.clone())
                    
                    preds_day_val = pytorch_model(X_day_val)

                    if preds_day_val.shape[0] != y_day_val_processed.shape[0]:
                        logger.error(f"Epoch {epoch+1}, Valid Day {i}: Mismatch shapes. Preds: {preds_day_val.shape}, Labels: {y_day_val_processed.shape}. Skipping.")
                        continue
                    if preds_day_val.numel() == 0:
                         continue


                    val_loss_item = criterion(preds_day_val.squeeze(), y_day_val_processed.squeeze())
                    
                    if not (torch.isnan(val_loss_item) or torch.isinf(val_loss_item)):
                        epoch_val_loss += val_loss_item.item()
                        processed_batches_valid +=1
                    else:
                        logger.warning(f"Epoch {epoch+1}, Valid Day {i}: NaN or Inf validation loss detected. Preds: {preds_day_val}, Labels: {y_day_val_processed}")


            avg_epoch_val_loss = epoch_val_loss / processed_batches_valid if processed_batches_valid > 0 else 0
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {avg_epoch_val_loss:.6f}")

            if scheduler:
                scheduler.step(avg_epoch_val_loss)
            if early_stopping: # early_stopping expects a loss (lower is better)
                if processed_batches_valid > 0: # Only step if val loss was computed
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
        pytorch_model.eval()
        all_predictions_list = []
        with torch.no_grad():
            for i in range(len(test_dataset)):
                X_day_test, y_day_test_actuals = test_dataset[i]
                
                current_date_for_day = test_dataset.unique_dates[i]
                day_specific_mask = test_dataset.multi_index.get_level_values('date') == current_date_for_day
                tickers_for_day = test_dataset.multi_index[day_specific_mask].get_level_values('ticker')

                X_day_test = X_day_test.to(device)
                preds_day_test = pytorch_model(X_day_test).squeeze().cpu().numpy()
                actuals_day_test = y_day_test_actuals.cpu().numpy()

                if len(tickers_for_day) != len(preds_day_test):
                    logger.error(f"Data mismatch for date {current_date_for_day}: {len(tickers_for_day)} tickers vs {len(preds_day_test)} preds. Skipping.")
                    continue
                
                for ticker_val, pred_score, actual_label in zip(tickers_for_day, preds_day_test, actuals_day_test):
                    all_predictions_list.append({
                        'date': current_date_for_day,
                        'ticker': ticker_val,
                        'prediction': pred_score,
                        'actual_return': actual_label 
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

if __name__ == "__main__":
    print("[main_multi_index.py] Script execution started from __main__.") # DIAGNOSTIC PRINT
    main()
    print("[main_multi_index.py] Script execution finished from __main__.") # DIAGNOSTIC PRINT
