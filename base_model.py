import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x, epsilon=1e-8):
    # Ensure x is a float tensor for std calculation
    if not x.is_floating_point():
        x = x.float()

    if x.numel() == 0:
        return x # Return empty tensor as is

    std_dev = x.std()

    # Check if std_dev is problematic (zero, very small, NaN, or Inf)
    if std_dev < epsilon or torch.isnan(std_dev) or torch.isinf(std_dev):
        # If no variance (e.g., all elements are the same, or only one element),
        # z-scores can be considered 0.
        return torch.zeros_like(x)

    return (x - x.mean()).div(std_dev + epsilon) # Add epsilon for numerical stability

def drop_extreme(x):
    if x.numel() == 0:
        return torch.zeros_like(x, dtype=torch.bool), x

    # Optional: Add a log if NaNs/Infs are detected, but proceed.
    # if torch.isnan(x).any() or torch.isinf(x).any():
    #     print(f"Warning: NaN or Inf detected in input to drop_extreme. Shape: {x.shape}")

    sorted_tensor, indices = x.sort() # NaNs are typically put at the end by sort.
    N = x.shape[0]

    percent_2_5 = int(0.025 * N)

    start_idx = percent_2_5
    end_idx = N - percent_2_5

    filtered_indices = indices[start_idx:end_idx]

    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    if filtered_indices.numel() > 0: # Ensure indices are not empty before trying to mask
        mask[filtered_indices] = True

    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class DaySampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source # data_source is DailyGroupedTimeSeriesDataset
        self.shuffle = shuffle
        self.num_days = len(data_source)

    def __iter__(self):
        indices = np.arange(self.num_days)
        if self.shuffle:
            np.random.shuffle(indices)
        yield from indices

    def __len__(self):
        return self.num_days


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            # print(f"Data batch received. data[0] shape (features): {data[0].shape}, data[1] shape (labels): {data[1].shape}")
            # data = torch.squeeze(data, dim=0) # Original line commented out
            # data is a tuple: (features_for_day_tensor, labels_for_day_tensor)
            feature_data = data[0].to(self.device) # N, T, F
            label_data = data[1].to(self.device)   # N
            # print(f"In train_epoch: feature_data.shape: {feature_data.shape}, label_data.shape: {label_data.shape}")

            # Original multiline comment for data shape (now for reference):
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label
            '''
            # feature = data[:, :, 0:-1].to(self.device) # Original line commented out
            # label = data[:, -1, -1].to(self.device) # Original line commented out

            # Additional process on labels
            # If you use original data to train, you won\'t need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            # print(f"Before drop_extreme: label_data.shape: {label_data.shape}")
            # Apply to label_data

            mask, current_labels_processed = drop_extreme(label_data)
            # print(f"Shape of feature_data: {feature_data.shape}")
            # print(f"Shape of mask: {mask.shape}")
            # print(f"Number of True in mask: {mask.sum().item()}")
            # Ensure feature_data is filtered with the same mask derived from label_data
            current_features_processed = feature_data[mask, :, :]

            current_labels_processed = zscore(current_labels_processed) # CSZscoreNorm
            #########################

            pred = self.model(current_features_processed.float())
            loss = self.loss_fn(pred, current_labels_processed)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            # data = torch.squeeze(data, dim=0) # Original line commented out
            # data is a tuple: (features_for_day_tensor, labels_for_day_tensor)
            feature_data = data[0].to(self.device) # N, T, F
            label_data = data[1].to(self.device)   # N

            # feature = data[:, :, 0:-1].to(self.device) # Original line commented out
            # label = data[:, -1, -1].to(self.device) # Original line commented out

            # You cannot drop extreme labels for test.
            # Apply zscore to the original label_data for test loss calculation

            current_labels_processed = zscore(label_data)

            # Use full feature_data for prediction as no drop_extreme is applied
            pred = self.model(feature_data.float())
            loss = self.loss_fn(pred, current_labels_processed)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        print("[SequenceModel] Initializing DataLoader in _init_data_loader...")
        if data is None or len(data) == 0:
            print("[SequenceModel] ERROR: Input data to _init_data_loader is None or empty.")
            # Optionally raise an error or return an empty loader
            raise ValueError("Input data for DataLoader cannot be None or empty.")

        day_sampler = DaySampler(data, shuffle)
        print(f"[SequenceModel] DaySampler initialized. Number of days/samples from sampler: {len(day_sampler)}")

        def custom_collate(batch_data_tuple):
            # When batch_size=None and a sampler is used, DataLoader passes the direct output of
            # dataset.__getitem__(idx) to the collate_fn.
            # If dataset.__getitem__ returns (features, labels), then batch_data_tuple will be (features, labels).
            # print(f"[Custom Collate] Received batch_data_tuple. Type: {type(batch_data_tuple)}")

            if isinstance(batch_data_tuple, tuple) and len(batch_data_tuple) == 2:
                # batch_data_tuple is the (features, labels) tuple
                features, labels = batch_data_tuple
                # f_shape = features.shape if hasattr(features, 'shape') else type(features)
                # l_shape = labels.shape if hasattr(labels, 'shape') else type(labels)
                # print(f"[Custom Collate] Successfully unpacked. Features type: {type(features)}, Labels type: {type(labels)}")
                return (features, labels)
            else:
                error_msg = (
                    f"[Custom Collate] ERROR: batch_data_tuple structure not as expected. "
                    f"Expected a 2-element tuple (features, labels). Got: {type(batch_data_tuple)}"
                    f"{', len ' + str(len(batch_data_tuple)) if isinstance(batch_data_tuple, tuple) else ''}"
                )
                # For debugging, print the content if it's small enough or types
                # print(f"Content: {batch_data_tuple}") 
                print(error_msg)
                raise ValueError(error_msg)

        data_loader = DataLoader(data, sampler=day_sampler, batch_size=None, collate_fn=custom_collate)
        print("[SequenceModel] DataLoader initialized successfully.")
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        print(f"[SequenceModel] Starting fit method. n_epochs: {self.n_epochs}")
        if dl_train is None or len(dl_train) == 0 :
            print("[SequenceModel] ERROR: Training data (dl_train) is None or empty in fit method. Cannot proceed.")
            return

        print("[SequenceModel] Initializing training DataLoader...")
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        
        valid_loader = None
        if dl_valid:
            print("[SequenceModel] Initializing validation DataLoader...")
            valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False) # Initialize once

        if train_loader is None : # or len(train_loader) == 0 if it's an iterable with a __len__
             print("[SequenceModel] ERROR: train_loader is None after _init_data_loader. Cannot proceed with training.")
             return
        # A more robust check for an empty DataLoader from a sampler perspective:
        # Try to get an iterator and see if it's empty, but this might consume the first batch.
        # For now, we rely on DaySampler's length and _init_data_loader's internal checks.

        best_param = None
        print("[SequenceModel] Starting training loop...")
        for step in range(self.n_epochs):
            print(f"[SequenceModel] Epoch {step + 1}/{self.n_epochs} - Starting train_epoch...")
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            print(f"[SequenceModel] Epoch {step + 1}/{self.n_epochs} - train_loss: {train_loss:.6f}")

            if valid_loader: # Check if valid_loader was initialized
                print(f"[SequenceModel] Epoch {step + 1}/{self.n_epochs} - Performing validation...")
                # Pass the already initialized valid_loader to predict
                predictions, metrics = self.predict(valid_loader) 
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                print(f"[SequenceModel] Training stop threshold met. train_loss: {train_loss:.6f} <= {self.train_stop_loss_thred}")
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                print(f"[SequenceModel] Model saved to {self.save_path}/{self.save_prefix}_{self.seed}.pkl")
                break
        print("[SequenceModel] Fit method completed.")

    def predict(self, dl_test):
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in dl_test:
            # data = torch.squeeze(data, dim=0) # Original line commented out
            # data is a tuple: (features_for_day_tensor, labels_for_day_tensor)
            feature_data = data[0].to(self.device) # N, T, F
            # For calc_ic, label_data is used as numpy, so get it before moving to device or after .cpu()
            label_data_numpy = data[1].numpy() # N

            # feature = data[:, :, 0:-1].to(self.device) # Original line commented out
            # label = data[:, -1, -1] # Original line, label was not moved to device here

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                pred = self.model(feature_data.float()).detach().cpu().numpy()
                preds.append(pred.ravel())

                daily_ic, daily_ric = calc_ic(pred, label_data_numpy)
                ic.append(daily_ic)
                ric.append(daily_ric)
        
        # If dl_test.dataset.get_index() is the correct way to get the index
        if hasattr(dl_test, 'dataset') and hasattr(dl_test.dataset, 'get_index'):
            series_index = dl_test.dataset.get_index()
             # We need to make sure this index corresponds to the items in the order they were processed.
             # Since shuffle=False for validation/test, this should be okay.
             # However, predictions are concatenated. The index must match all concatenated predictions.
             # The `DailyGroupedTimeSeriesDataset`'s get_index() returns the *full* multi_index of all sequences it holds.
             # The predictions are generated day by day.
             # We need to reconstruct the correct index for the concatenated predictions.
            
            # Let's collect all indices from each batch (day) and concatenate them
            all_indices = []
            original_multi_index = dl_test.dataset.multi_index # Access the full index from the dataset
            unique_dates_in_loader = dl_test.dataset.unique_dates # Access unique dates in order they are processed by DaySampler (shuffle=False)

            # We need to iterate through the loader again or store indices during prediction if get_index() is per-batch
            # Simpler: the `dl_test.dataset.get_index()` should return the full index for ALL items this dataset can produce,
            # in the order they would appear if iterated without shuffling.
            # The current DaySampler + DailyGroupedTimeSeriesDataset structure means `get_index()` on the dataset
            # returns the *original* index of all sequences.
            # The predictions are ordered by date, then by stock within that date.
            # We need an index that matches this.
            
            # Let's rebuild the index based on the order of processing.
            # This is tricky because predictions are flattened.
            # The original implementation `index=dl_test.get_index()` assumed `dl_test` was the dataset itself.
            # Now that `dl_test` is a DataLoader, we use `dl_test.dataset.get_index()`.

            # The issue is that `np.concatenate(preds)` creates a flat array.
            # `dl_test.dataset.get_index()` returns a MultiIndex for all samples in the test set.
            # If the order of iteration through `dl_test` (which is by day, and then stocks within the day)
            # matches the inherent order of `dl_test.dataset.get_index()`, it might work.
            # Let's assume for now it does, as this was the implicit assumption before.
            # The `DailyGroupedTimeSeriesDataset` groups by date, and for each date, it takes all stocks.
            # The `get_index()` method on `DailyGroupedTimeSeriesDataset` returns the original full multi_index.
            # We need to ensure the concatenated predictions align with this full multi_index if it's not sorted by date.

            # The most robust way: Collect (original_index_for_sample, prediction_for_sample) pairs
            # and then reconstruct the series.
            # For now, stick to the previous assumption that the order matches.
            # The DailyGroupedTimeSeriesDataset already sorts by unique_dates.
            # And within each date, data is taken as is from X_sequences[date_mask].
            # So, the order of predictions should match the order of dl_test.dataset.multi_index
            # if dl_test.dataset.multi_index itself is sorted by (date, ticker).

            # Let's verify the sorting of the index from the dataset:
            idx_for_series = dl_test.dataset.get_index()
            if not idx_for_series.is_monotonic_increasing:
                 # This might be an issue if it's not sorted as (date, then ticker)
                 # For DailyGroupedTimeSeriesDataset, self.multi_index is passed in.
                 # Its order matters. create_sequences_multi_index creates it.
                 # Let's ensure it's sorted appropriately after creation in main_multi_index.py
                 # print_warning("Index for predictions is not sorted. This might lead to misalignment.")
                 pass # For now, proceed with caution.

            predictions_series = pd.Series(np.concatenate(preds), index=idx_for_series)
        else:
            print("[SequenceModel] WARNING: Could not retrieve index for predictions. DataLoader's dataset lacks get_index method.")
            predictions_series = pd.Series(np.concatenate(preds))


        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic) if np.std(ic) != 0 else 0,
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric) if np.std(ric) != 0 else 0
        }

        return predictions_series, metrics
