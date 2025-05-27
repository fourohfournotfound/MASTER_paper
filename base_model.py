import numpy as np
import pandas as pd
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler # Changed Sampler to Dataset for TensorDataset

def calc_ic(pred, label):
    # Ensure inputs are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    
    pred = pred.flatten()
    label = label.flatten()

    df = pd.DataFrame({'pred':pred, 'label':label})
    # Drop rows where label might be NaN after potential processing,
    # or if pred is NaN (though less likely for model output)
    df.dropna(subset=['label', 'pred'], inplace=True)
    if df.shape[0] < 2: # Not enough data points for correlation
        return 0.0, 0.0

    # Check for constant series after NaN drop
    if df['pred'].nunique() <= 1 or df['label'].nunique() <= 1:
        return 0.0, 0.0 # Or handle as an error/warning

    ic = df['pred'].corr(df['label'], method='pearson')
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic if not np.isnan(ic) else 0.0, ric if not np.isnan(ric) else 0.0

def zscore(x: torch.Tensor):
    # Ensure we don't try to zscore an empty or all-NaN tensor after filtering
    if x.numel() == 0 or torch.isnan(x).all():
        return x # Return as is, or handle error
    mean = x.mean()
    std = x.std()
    if std == 0: # Avoid division by zero
        return x - mean # Just center if std is zero
    return (x - mean) / std

def drop_extreme(x: torch.Tensor):
    # Ensure x is 1D
    x_1d = x.flatten()
    if x_1d.numel() == 0:
        return torch.empty_like(x_1d, dtype=torch.bool), x_1d

    sorted_tensor, indices = torch.sort(x_1d)
    N = x_1d.shape[0]
    percent_2_5 = int(0.025*N)
    
    # Handle cases where N is too small for 2.5% drop from both ends
    if N <= 1 or percent_2_5 * 2 >= N : # If dropping 2.5% from each end removes everything or more
        # Don't drop anything if the tensor is too small to meaningfully apply the rule
        mask = torch.ones_like(x_1d, device=x.device, dtype=torch.bool)
        return mask, x_1d[mask]

    filtered_indices = indices[percent_2_5 : N - percent_2_5] # Corrected slicing
    
    mask = torch.zeros_like(x_1d, device=x.device, dtype=torch.bool)
    if filtered_indices.numel() > 0: # Ensure filtered_indices is not empty
      mask[filtered_indices] = True
    return mask, x_1d[mask]


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source) # Added: Call to superclass constructor
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        # Assuming data_source has a get_index() method that returns a pandas MultiIndex like StockDataset
        idx = self.data_source.get_index()
        self.daily_count = pd.Series(index=idx).groupby(idx.get_level_values('datetime')).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            # Shuffle the order of days, not samples within a day for typical financial TS
            day_indices = np.arange(len(self.daily_count))
            np.random.shuffle(day_indices)
            for i in day_indices:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx_start, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx_start, idx_start + count)

    def __len__(self):
        # This should return the number of batches (days), not total samples
        return len(self.daily_count)


class SequenceModel():
    def __init__(self, n_epochs, lr, batch_size=64, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size # Added batch_size for standard DataLoader
        self.device = torch.device(f"cuda:{GPU}" if GPU is not None and torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred # This is a loss threshold

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True #type: ignore
        self.fitted = -1 # Tracks epochs trained or status

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix
        # Ensure save_path directory exists
        import os
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if torch.sum(mask).item() == 0: # No valid labels
            return torch.tensor(0.0, device=pred.device, requires_grad=True) # Or handle as appropriate
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for feature_batch, label_batch in data_loader: # Adapted for (feature, label) from DataLoader
            feature = feature_batch.to(self.device).float()
            label_original = label_batch.to(self.device).float().squeeze(-1) # Assuming label_batch might be (batch, 1)

            # Process labels per batch: drop_extreme then zscore
            # Ensure label is 1D for drop_extreme and zscore as implemented
            if label_original.numel() > 0:
                mask_extreme, label_processed = drop_extreme(label_original)
                if label_processed.numel() > 0: # Ensure not empty after dropping extremes
                    feature = feature[mask_extreme, :, :] # Filter features based on labels
                    label = zscore(label_processed)
                else: # All labels dropped
                    continue # Skip batch if no valid labels remain
            else: # Empty label tensor
                continue

            if label.numel() == 0: # Double check after processing
                continue

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN or Inf loss detected during training. Skipping batch.")
                continue

            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        
        if not losses: # Handle case where all batches were skipped
            return float('nan') # Or some indicator of no training done
        return float(np.mean(losses))

    def test_epoch(self, data_loader): # Used for validation loss during fit
        self.model.eval()
        losses = []

        with torch.no_grad():
            for feature_batch, label_batch in data_loader:
                feature = feature_batch.to(self.device).float()
                label_original = label_batch.to(self.device).float().squeeze(-1)

                # Process labels per batch: zscore only for test/validation
                if label_original.numel() > 0:
                    label = zscore(label_original)
                else:
                    continue # Skip if empty

                if label.numel() == 0:
                    continue
                
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    losses.append(loss.item())
                else:
                    print("Warning: NaN or Inf loss detected during testing/validation.")


        if not losses:
            return float('nan')
        return float(np.mean(losses))

    def _init_data_loader(self, data_set: Dataset, shuffle=True, drop_last=True):
        # data_set is now expected to be a torch.utils.data.Dataset object
        return DataLoader(data_set, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

    def load_param(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = f'Loaded from {param_path}'
            print(f"Model parameters loaded from {param_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {param_path}")
            raise
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            raise


    def fit(self, train_data: Dataset, valid_data: Dataset = None):
        # Ensure model is initialized
        if self.model is None: self.init_model()
        if self.train_optimizer is None: self.init_model() # Redundant if init_model called, but safe

        train_loader = self._init_data_loader(train_data, shuffle=True, drop_last=True)
        if valid_data:
            valid_loader = self._init_data_loader(valid_data, shuffle=False, drop_last=False)

        best_valid_loss = float('inf')
        best_param = None

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            
            log_msg = f"Epoch {step + 1}/{self.n_epochs}, Train Loss: {train_loss:.6f}"

            if valid_data:
                # Original predict gives ICs, test_epoch gives loss. For validation loop, usually loss is primary.
                # If original fit method called predict for metrics, let's align.
                # The original fit() example showed predict(dl_valid).
                predictions_val, metrics_val = self.predict(valid_data) # This will use its own loader
                valid_loss = self.test_epoch(valid_loader) # More direct for validation loss monitoring

                log_msg += f", Valid Loss: {valid_loss:.6f}, Valid IC: {metrics_val.get('IC', float('nan')):.4f}, Valid RIC: {metrics_val.get('RIC', float('nan')):.4f}"
                print(log_msg)
                
                current_metric_for_saving = valid_loss # Or could be metrics_val['IC'] if optimizing for that
                if current_metric_for_saving < best_valid_loss: # Assuming lower loss is better
                    best_valid_loss = current_metric_for_saving
                    best_param = copy.deepcopy(self.model.state_dict())
                    print(f"New best validation loss: {best_valid_loss:.6f}. Saving model.")
                    if self.save_path and self.save_prefix:
                         torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed if self.seed is not None else "model"}_best.pkl')

            else:
                print(log_msg)
                # If no validation, save based on train_stop_loss_thred if provided
                if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                    print(f"Training loss {train_loss:.6f} reached threshold {self.train_stop_loss_thred}. Saving model.")
                    best_param = copy.deepcopy(self.model.state_dict())
                    if self.save_path and self.save_prefix:
                        torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed if self.seed is not None else "model"}_final.pkl')
                    break # Stop training
            
            self.fitted = step + 1 # Number of epochs completed

        if best_param is None and self.save_path and self.save_prefix: # Save last model if no best was found/saved
             torch.save(self.model.state_dict(), f'{self.save_path}/{self.save_prefix}_{self.seed if self.seed is not None else "model"}_last_epoch.pkl')
             print("Saved model from last epoch as no improvement or threshold condition met for 'best' model.")
        elif best_param:
             print("Best model saved during training.")


    def predict(self, test_data: Dataset):
        if self.fitted == -1 and not (hasattr(self.model, 'state_dict') and any(self.model.state_dict())):
            # A bit of a loose check if model might have been loaded without setting self.fitted string.
            # A better way is to ensure load_param sets self.fitted appropriately.
            print("Warning: Model might not be fitted or loaded. Predictions may be random.")
            if self.model is None: self.init_model() # Initialize if not even done that

        test_loader = self._init_data_loader(test_data, shuffle=False, drop_last=False)

        preds_list = []
        labels_list = [] # Collect actual labels to compute IC/RIC correctly batch-wise or overall

        self.model.eval()
        with torch.no_grad():
            for feature_batch, label_batch in test_loader:
                feature = feature_batch.to(self.device).float()
                label_original = label_batch.to(self.device).float().squeeze(-1)
                
                # For prediction, the original applied zscore to label for consistency in loss calc
                # but for IC/RIC, raw or consistently scaled labels are better.
                # Here, we just collect raw labels associated with predictions.
                # The `calc_ic` function will handle its own requirements.

                pred = self.model(feature).cpu() # Bring preds to CPU
                preds_list.append(pred.numpy())
                labels_list.append(label_original.cpu().numpy())


        predictions_np = np.concatenate(preds_list).flatten()
        actual_labels_np = np.concatenate(labels_list).flatten()
        
        # The original calc_ic was daily. Here we do it over the whole test set.
        # For multi-index data, if dl_test had get_index(), one could try to group by date.
        # For now, overall IC/RIC.
        ic, ric = calc_ic(predictions_np, actual_labels_np)

        metrics = {
            'IC': ic,
            'ICIR': np.nan, # ICIR requires daily ICs, not directly computable here
            'RIC': ric,
            'RICIR': np.nan # RICIR also requires daily RICs
        }
        
        # The original returned pd.Series. If test_data is TensorDataset, it has no index.
        # For now, return numpy array of predictions.
        # If an index is available from test_data (e.g. if it were a custom Dataset subclass), use it.
        return predictions_np, metrics
