import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import time

from master import MASTERModel
from base_model import SequenceModel, DailyBatchSamplerRandom, calc_ic, zscore, drop_extreme # Add other necessary imports from base_model

# This specialized dataset is for use with DailyBatchSamplerRandom
class DailyGroupedTimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, sequence_length=8):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.sequence_length = sequence_length

        self.daily_feature_tensors = [] # List of tensors, each tensor is (num_stocks_on_day, sequence_length, num_features)
        self.daily_label_tensors = [] # List of tensors, each tensor is (num_stocks_on_day)
        self.daily_indices = [] # List of pandas MultiIndex, for each day
        self.daily_original_indices = [] # Stores original (ticker, date) for items in each daily batch

        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date')):
            df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]), level='date')

        # Ensure the DataFrame is sorted by date, then ticker for consistent processing
        df = df.sort_index(level=['date', 'ticker'])

        # Group by date first
        for date, daily_group in df.groupby(level='date'):
            day_features_list = []
            day_labels_list = []
            day_indices_list = []

            # For each stock on that day, find its sequence
            for ticker, stock_data_on_day in daily_group.groupby(level='ticker'):
                # Get the full history for this stock to form the sequence
                # This requires having the full df available, or at least enough history
                # This approach is tricky with a simple df split.
                # A more robust way is to generate all possible (sequence, label) pairs first,
                # then group them by the date of the label.

                # Let's pre-generate all sequences and their corresponding (ticker, date) for the label
                # This is done outside and then passed to this dataset, or this dataset does it once.
                pass # This part will be complex and needs careful implementation.
                     # The TimeSeriesDataset above is a better starting point for sequence generation.
                     # We then need to group those sequences by their end date for DailyBatchSamplerRandom.


        # --- Alternative approach for DailyGroupedTimeSeriesDataset ---
        # 1. Generate all (sequence, label, (ticker, label_date)) tuples using a logic similar to TimeSeriesDataset
        all_sequences = []
        all_labels_for_seq = []
        all_indices_for_seq = [] # (ticker, date_of_label)

        # Corrected date parsing for MultiIndex
        if isinstance(df.index.levels[1], pd.Index) and not pd.api.types.is_datetime64_any_dtype(df.index.levels[1].dtype):
            df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]), level='date')
        elif isinstance(df.index.levels[1], pd.DatetimeIndex): # Already datetime
            pass
        else: # Fallback for other cases, might need more specific handling
             df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]), level='date')


        for ticker, group in df.groupby(level='ticker'):
            features_np = group[self.feature_cols].values
            labels_np = group[self.label_col].values

            for i in range(len(group) - self.sequence_length +1): # +1 to ensure we can get a label for the sequence
                if i + self.sequence_length <= len(features_np): # Ensure there's enough data for a full sequence and its label
                    seq = features_np[i : i + self.sequence_length]
                    # Label is for the day *after* the sequence ends, but pct_change().shift(-1) already handles this.
                    # So, the label corresponds to the last day of the sequence.
                    label = labels_np[i + self.sequence_length - 1]
                    idx_tuple = group.index[i + self.sequence_length - 1] # (ticker, date_of_label)

                    all_sequences.append(seq)
                    all_labels_for_seq.append(label)
                    all_indices_for_seq.append(idx_tuple)

        # Now group these sequences by the date of their label
        # Create a temporary DataFrame to help with grouping
        temp_df_for_grouping = pd.DataFrame({
            'sequence': all_sequences,
            'label': all_labels_for_seq,
            'ticker': [idx[0] for idx in all_indices_for_seq],
            'datetime': [idx[1] for idx in all_indices_for_seq] # This is the date of the label
        })

        # This index is what DailyBatchSamplerRandom will use
        self._internal_index = pd.MultiIndex.from_tuples(all_indices_for_seq, names=['ticker', 'datetime'])


        if temp_df_for_grouping.empty:
            print("Warning: No sequences were generated. Check sequence_length and data size.")
            self._internal_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['ticker', 'datetime']) # Empty MultiIndex
            # Ensure daily_data, daily_labels, daily_indices are initialized as empty lists
            self.daily_feature_tensors = []
            self.daily_label_tensors = []
            self.daily_original_indices = [] # To store the original (ticker, date) for each item in the batch
            return


        # Group by the label's date
        for date_of_label, group in temp_df_for_grouping.groupby('datetime'):

            # print(f"--- Processing day: {date_of_label} ---")
            # print(f"  Number of items in group for this day: {len(group)}")

            sequences_list = group['sequence'].values
            # print(f"  group['sequence'].values type: {type(sequences_list)}")
            # print(f"  Number of sequences for this day: {len(sequences_list)}")
            # if len(sequences_list) > 0 and isinstance(sequences_list[0], np.ndarray):
            #     print(f"  Shape of first sequence in sequences_list: {sequences_list[0].shape}")
            # elif len(sequences_list) > 0:
            #     print(f"  Type of first sequence in sequences_list: {type(sequences_list[0])}")

            stacked_sequences_np_exists = False # Flag to check if stacked_sequences_np was successfully created
            try:
                # Ensure sequences_list is not empty before stacking
                if len(sequences_list) == 0:
                    # print(f"WARNING: sequences_list is empty for day {date_of_label}. Creating empty feature tensor.")
                    sequences_on_day = torch.empty(0, self.sequence_length, len(self.feature_cols), dtype=torch.float32)
                    stacked_sequences_np = np.array([]) # Define for the flag logic, though not strictly needed if sequences_on_day is empty
                else:
                    stacked_sequences_np = np.stack(sequences_list)
                    stacked_sequences_np_exists = True
                    # print(f"  Shape of stacked_sequences_np (from features): {stacked_sequences_np.shape}")
                    sequences_on_day = torch.tensor(stacked_sequences_np, dtype=torch.float32)
            except ValueError as e:
                print(f"WARNING: Error stacking sequences for day {date_of_label}: {e}. Creating empty feature tensor for this day.")
                sequences_on_day = torch.empty(0, self.sequence_length, len(self.feature_cols), dtype=torch.float32)
                stacked_sequences_np = np.array([]) # Define for flag logic

            labels_array = group['label'].values
            # print(f"  group['label'].values type: {type(labels_array)}")
            # print(f"  group['label'].values shape: {labels_array.shape}")
            # if labels_array.ndim == 1 and len(labels_array) > 0:
            #     print(f"  First label value: {labels_array[0]}")

            if sequences_on_day.shape[0] == 0:
                 labels_on_day = torch.empty(0, dtype=torch.float32)
                 if len(labels_array) > 0 and len(sequences_list) > 0 and not stacked_sequences_np_exists:
                     # This case means np.stack failed for non-empty sequences_list
                     pass # sequences_on_day is already empty, labels_on_day is now also empty
                 elif len(labels_array) > 0 and len(sequences_list) == 0:
                     # This case means sequences_list was empty from the start
                     pass # sequences_on_day is already empty, labels_on_day is now also empty
            else: # sequences_on_day is not empty
                 labels_on_day = torch.tensor(labels_array, dtype=torch.float32)
                 if sequences_on_day.shape[0] != labels_on_day.shape[0]:
                     print(f"WARNING: Mismatch in number of samples for day {date_of_label}. Features: {sequences_on_day.shape[0]}, Labels: {labels_on_day.shape[0]}. Adjusting both to empty for this day.")
                     labels_on_day = torch.empty(0, dtype=torch.float32)
                     sequences_on_day = torch.empty(0, self.sequence_length, len(self.feature_cols), dtype=torch.float32)

            # print(f"  Shape of created sequences_on_day: {sequences_on_day.shape} (Expected N, {self.sequence_length}, {len(self.feature_cols)}) ")
            # print(f"  Shape of created labels_on_day: {labels_on_day.shape} (Expected N)")
            # print(f"--- End processing day: {date_of_label} ---")

            # Stack sequences for this day: (num_stocks, sequence_length, num_features)
            # Stack labels for this day: (num_stocks)
            # Keep original indices for this day

            # Convert list of sequences (each is numpy array) to a single numpy array, then to tensor
            # sequences_on_day = np.array(group['sequence'].tolist()) # This might create ragged array if not careful

            # Ensure all sequences in a batch have the same shape before stacking
            # This should be guaranteed by the sequence generation logic
            sequences_on_day = torch.tensor(np.stack(group['sequence'].values), dtype=torch.float32)
            labels_on_day = torch.tensor(group['label'].values, dtype=torch.float32)

            self.daily_feature_tensors.append(sequences_on_day)
            self.daily_label_tensors.append(labels_on_day)
            # Store the (ticker, date_of_label) for items in this batch
            self.daily_original_indices.append(pd.MultiIndex.from_frame(group[['ticker', 'datetime']]))


    def __len__(self):
        # This should be the number of unique days for which we have data
        return len(self.daily_feature_tensors)

    def __getitem__(self, idx): # idx here is the index of the day batch
        # Returns all data for a specific day (which is a batch for DailyBatchSamplerRandom)
        # (feature_batch_for_day, label_batch_for_day)
        # feature_batch_for_day: (num_stocks_on_day, sequence_length, num_features)
        # label_batch_for_day: (num_stocks_on_day)
        return self.daily_feature_tensors[idx], self.daily_label_tensors[idx]

    def get_index(self):
        # This index is used by DailyBatchSamplerRandom to know how many items (stocks) are on each day.
        # It should be a MultiIndex of (ticker, datetime) for all individual samples (sequences) before daily grouping.
        return self._internal_index


def load_and_preprocess_data(csv_path, sequence_length=8):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.set_index(['ticker', 'date'], inplace=True)
    df.sort_index(inplace=True)

    # Define features and label

    # Drop 'lastupdated' if it exists and is not used, or convert it
    if 'lastupdated' in df.columns:
        df = df.drop(columns=['lastupdated'])



    # Calculate next-day returns using closeadj
    df['label'] = df.groupby(level='ticker')['closeadj'].pct_change(1).shift(-1)

    # Handle potential infinities from pct_change if closeadj was 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
\
    # Calculate mean market return
    print("Calculating mean market return...")
    if 'closeadj' not in df.columns:
        raise ValueError("'closeadj' column is required to calculate mean market return but not found.")
    df_pivot = df['closeadj'].unstack(level='ticker')
    daily_returns = df_pivot.pct_change()
    mean_market_return = daily_returns.mean(axis=1)
    mean_market_return.name = 'mean_market_return'
    df = df.join(mean_market_return, on='date')
    df['mean_market_return'] = df['mean_market_return'].fillna(0)
    print("Mean market return calculated and added as a feature.")

    # Define feature_cols dynamically
    feature_cols = [col for col in df.columns if col != 'label']
    if not feature_cols:
        raise ValueError("No feature columns found after processing.")
\
    print(f"Checking for NaNs in feature columns before fillna. Sum of NaNs: {df[feature_cols].isnull().sum().sum()}")
    # Fill NaNs in all feature columns. Using 0 for simplicity.
    # Consider ffill or other strategies if 0 is not appropriate for all features.
    for col in feature_cols:
        if df[col].isnull().any():
            print(f"NaNs found in feature column '{col}'. Count: {df[col].isnull().sum()}. Filling with 0.")
            df[col] = df[col].fillna(0)
    print(f"Checking for NaNs in feature columns after fillna. Sum of NaNs: {df[feature_cols].isnull().sum().sum()}")
    print(f"Feature_cols (first 5): {feature_cols[:5]}... (Total: {len(feature_cols)})")

    # Drop rows where the label is NaN
    df.dropna(subset=['label'], inplace=True)
    print(f"Shape after label NaN drop: {df.shape}")



    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing (NaN handling). Check data quality or sequence length.")

    print(f"Data loaded. Shape: {df.shape}")
    # Splitting data (example: 70% train, 15% validation, 15% test)
    # This needs to be done carefully for time series data, respecting chronological order.
    # And also ensuring DailyBatchSamplerRandom gets a continuous block of dates for each set.

    unique_dates = df.index.get_level_values('date').unique().sort_values()
    n_unique_dates = len(unique_dates)
    if n_unique_dates == 3:
        print("Adjusting split for 3 unique dates: 1 train, 1 valid, 1 test.")
        train_dates = unique_dates[:1]
        valid_dates = unique_dates[1:2]
        test_dates = unique_dates[2:3]
    elif n_unique_dates < 3:
        raise ValueError(f"Not enough unique dates to split into train, validation, and test sets. Found: {n_unique_dates}")
    else: # General case for n_unique_dates > 3
        train_end_idx = int(n_unique_dates * 0.7)
        # Ensure train gets at least one date
        if n_unique_dates > 0: train_end_idx = max(1, train_end_idx)

        valid_start_idx = train_end_idx
        valid_end_idx = int(n_unique_dates * 0.85)

        # If valid_end_idx is not greater than valid_start_idx, try to make valid set have at least one day
        if valid_end_idx <= valid_start_idx and valid_start_idx < n_unique_dates:
            valid_end_idx = valid_start_idx + 1

        # Cap valid_end_idx at the total number of unique dates
        valid_end_idx = min(valid_end_idx, n_unique_dates)

        # Ensure valid_start_idx does not exceed n_unique_dates
        valid_start_idx = min(valid_start_idx, n_unique_dates)


        train_dates = unique_dates[:train_end_idx]
        valid_dates = unique_dates[valid_start_idx:valid_end_idx]
        test_dates = unique_dates[valid_end_idx:]

    df_train = df[df.index.get_level_values('date').isin(train_dates)]
    df_valid = df[df.index.get_level_values('date').isin(valid_dates)]
    df_test = df[df.index.get_level_values('date').isin(test_dates)]

    print(f"Train shape: {df_train.shape}, Valid shape: {df_valid.shape}, Test shape: {df_test.shape}")
    if df_train.empty or df_valid.empty or df_test.empty:
        print("Warning: One or more data splits are empty. Adjust split ratios or check data.")


    # Create datasets
    # The DailyGroupedTimeSeriesDataset is what we need for the existing sampler
    ds_train = DailyGroupedTimeSeriesDataset(df_train, feature_cols, 'label', sequence_length)
    ds_valid = DailyGroupedTimeSeriesDataset(df_valid, feature_cols, 'label', sequence_length)
    ds_test = DailyGroupedTimeSeriesDataset(df_test, feature_cols, 'label', sequence_length)

    d_feat = len(feature_cols)

    return ds_train, ds_valid, ds_test, d_feat


def main():
    parser = argparse.ArgumentParser(description='Train MASTER model with multi-index CSV data.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the multi-index CSV file.')
    parser.add_argument('--sequence_length', type=int, default=8, help='Length of the input sequence.')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model.')
    parser.add_argument('--t_nhead', type=int, default=4, help='Number of heads in Temporal Attention.')
    parser.add_argument('--s_nhead', type=int, default=2, help='Number of heads in Stock Attention.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta for Gate module.') # Default from main.py for csi800 like
    parser.add_argument('--n_epoch', type=int, default=1, help='Number of training epochs.') # Low for quick test
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--train_stop_loss_thred', type=float, default=0.95, help='Training stop loss threshold.')
    parser.add_argument('--save_path', type=str, default='model_multi', help='Path to save models.')
    parser.add_argument('--model_prefix', type=str, default='master_csv_model', help='Prefix for saved model files.')


    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Gate input indices - this needs to be adapted.
    # The original model used specific alpha features for gating.
    # For now, let's assume we are not using the complex gating or using all features.
    # If d_feat is small (e.g., 7), these original indices (158, 221) are out of bounds.
    # Option 1: Don't use gate, or modify Gate/MASTER to not require separate gate_input features from x.
    # Option 2: Select a subset of our 7 features for the gate.
    # For simplicity, let's make gate_input_start_index = 0 and gate_input_end_index = d_feat (i.e. use all features for gate decision)
    # This might not be what the original Gate was designed for but makes it runnable.
    # The MASTER model uses x[:, -1, gate_start:gate_end] for gate input.
    # And x[:, :, :gate_start] for main features. This implies gate features are appended.
    # This needs a more careful redesign if we want to replicate the original paper's gating.
    # For now, let's set gate_input_start_index to be d_feat, and gate_input_end_index to also be d_feat,
    # effectively making the gate input empty or requiring modification to the MASTER model.
    # A simpler approach for now: modify MASTER to take d_feat and handle gating internally if needed,
    # or make the gate operate on a subset of the main d_feat features.

    # Let's assume the gate will operate on the *last* few features of the `d_feat` input if used.
    # Or, we can set gate_input_start_index = d_feat and gate_input_end_index = d_feat,
    # which would mean the gate_input slice is empty. The Gate module would then get an empty tensor.
    # This will likely cause an error.

    # Simplest first step: The `MASTER` class in `master.py` slices features like:
    # src = x[:, :, :self.gate_input_start_index]
    # gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
    # If gate_input_start_index = d_feat, then src takes all features, and gate_input is empty.
    # This requires the Gate module to handle empty input or be modified.

    # Let's try to make the gate use *all* the features for its decision, by setting gate_input_start_index = 0
    # and gate_input_end_index = d_feat. This means `src` in MASTER will be x[:,:,:0] (empty!)
    # This is also problematic.

    # The original paper seems to have base features (d_feat) and then additional features for the gate.
    # Our current setup only has d_feat features.
    # For now, let's define gate_input_start_index = 0, gate_input_end_index = args.d_feat
    # This means the `src` in MASTER will be `x[:, :, :0]` which is empty.
    # This implies the `MASTER` model structure needs to be adapted if the input features don't separate base and gate features.

    # TEMPORARY: For the model to run, let's assume gate_input_start_index is where non-gated features end.
    # And the rest are for the gate. If we only have `d_feat` total, and want to use *all* for main model,
    # and also *all* for gate, the current MASTER slicing won't work directly.

    # Let's assume d_feat is the total number of features in the input tensor `x`.
    # The MASTER class expects `x` to contain both main features and gate features.
    # If `gate_input_start_index` is, say, 5 (out of 7 features), then features 0-4 are main, 5-6 are for gate.
    # Let's set gate_input_start_index = d_feat (actual loaded)
    # and gate_input_end_index = d_feat. This means the gate gets an empty input.
    # This will require modifying the Gate or MASTER class.

    # For now, to make progress, we will set them such that the gate is effectively bypassed or uses all features.
    # This part is CRITICAL and needs to align with the MASTER model's assumptions.
    # The provided `main.py` has d_feat=158, gate_start=158, gate_end=221.
    # This means the first 158 features are for the main model path, and features 158-220 are for the gate.
    # The input `x` to MASTER.forward() would have 221 features in total.

    # With our 7 features, we can't do this directly.
    # Easiest path: Modify `MASTER` to not use the feature_gate if `d_gate_input` is 0.
    # Or, we can set `gate_input_start_index = args.d_feat` and `gate_input_end_index = args.d_feat`.
    # This makes `self.d_gate_input = 0`. The `Gate` class needs to handle this.

    # Let's assume for now:
    # The `d_feat` argument to MASTERModel will be the number of features for the main path.
    # The gate features will be separate. If we don't have separate gate features,
    # we need to adjust.

    # For now, let's assume all loaded features are "main" features.
    # So, `d_feat_from_data` is what `load_and_preprocess_data` returns.
    # `gate_input_start_index` will be `d_feat_from_data`.
    # `gate_input_end_index` will also be `d_feat_from_data` (meaning no *additional* features for gate from input `x`).
    # The `Gate` module in `master.py` takes `d_input` which is `gate_input_end_index - gate_input_start_index`.
    # If this is 0, `Gate` will try to create `nn.Linear(0, d_output)`. This is an error.

    # --> We MUST modify the MASTER class or Gate if we don't provide distinct gate features in `x`.
    # Simplification: Let's assume the `d_feat` passed to `MASTERModel` is the total number of features available.
    # And the `Gate` will operate on these same features.
    # So, in `MASTER.forward`:
    # `src = x` (all features)
    # `gate_input = x[:, -1, :]` (all features of the last day for gate decision)
    # This requires `gate_input_start_index = 0` and `gate_input_end_index = d_feat_from_data`
    # Then `self.d_gate_input` in MASTER becomes `d_feat_from_data`.
    # And `src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)` will apply gate to all features.

    print("Loading and preprocessing data...")
    ds_train, ds_valid, ds_test, d_feat_from_data = load_and_preprocess_data(args.csv_path, args.sequence_length)

    # If any dataset is empty (e.g. due to small CSV and sequence length), stop.
    if len(ds_train) == 0 or len(ds_valid) == 0 or len(ds_test) == 0:
        print("One of the datasets (train, valid, test) is empty after processing. Exiting.")
        print(f"Train length: {len(ds_train)}, Valid length: {len(ds_valid)}, Test length: {len(ds_test)}")
        # Further check if the internal lists of DailyGroupedTimeSeriesDataset are populated
        if hasattr(ds_train, 'daily_feature_tensors'): # Check if attribute exists
             print(f"ds_train.daily_feature_tensors length: {len(ds_train.daily_feature_tensors)}")
        if hasattr(ds_valid, 'daily_feature_tensors'):
             print(f"ds_valid.daily_feature_tensors length: {len(ds_valid.daily_feature_tensors)}")
        if hasattr(ds_test, 'daily_feature_tensors'):
            print(f"ds_test.daily_feature_tensors length: {len(ds_test.daily_feature_tensors)}")
        return

    # For the MASTER model, d_feat is the number of features for the main path.
    # The 'mean_market_return' feature (which was the last one added to feature_cols)
    # will be used as the single input for the gate.
    # d_feat_from_data is the total number of features including 'mean_market_return'.
    # The index of 'mean_market_return' is d_feat_from_data - 1.
    effective_gate_input_start_index = d_feat_from_data - 1
    effective_gate_input_end_index = d_feat_from_data # Slice will be [d_feat_from_data - 1:d_feat_from_data], yielding 1 feature
    print(f"Gate input will use feature index {effective_gate_input_start_index} (the 'mean_market_return').")
    print(f"Total features (d_feat_from_data) for the model: {d_feat_from_data}")

    print(f"Initializing model with d_feat={d_feat_from_data}, d_model={args.d_model}...")
    model = MASTERModel(
        d_feat=d_feat_from_data,  # Features for the main model path
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        T_dropout_rate=args.dropout,
        S_dropout_rate=args.dropout,
        # These gate indices are relative to the input `x` of MASTER.forward
        gate_input_start_index=effective_gate_input_start_index, # All features are used by gate
        gate_input_end_index=effective_gate_input_end_index,     # All features are used by gate
        beta=args.beta,
        n_epochs=args.n_epoch,
        lr=args.lr,
        GPU=args.gpu,
        seed=args.seed,
        train_stop_loss_thred=args.train_stop_loss_thred,
        save_path=args.save_path,
        save_prefix=args.model_prefix
    )

    print("Model Initialized.")

    # Training
    if args.n_epoch > 0:
        print("Starting training...")
        start_time = time.time()
        model.fit(ds_train, ds_valid)
        print(f"Training finished. Time: {time.time() - start_time:.2f}s")
    else:
        print("Skipping training as n_epoch is 0.")
        # If a pre-trained model should be loaded, add logic here
        # model.load_param(f'{args.save_path}/{args.model_prefix}_{args.seed}.pkl')


    # Testing
    print("Starting testing...")
    predictions, metrics = model.predict(ds_test)
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save predictions if needed
    # predictions.to_csv(f'{args.save_path}/{args.model_prefix}_predictions_{args.seed}.csv')
    print("Predictions series (first 5):")
    print(predictions.head())


if __name__ == '__main__':
    main()

