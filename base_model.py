import numpy as np
import pandas as pd
import copy
import logging
import sys
from pathlib import Path
import time
import psutil

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from data_leakage_fixes import (
    zscore_per_day, 
    robust_loss_with_outlier_handling,
    apply_feature_clipping
)

# Setup basic logging for this module if not already configured globally
logger = logging.getLogger(__name__)
# Ensure a handler is configured for the logger if running this module standalone or if not configured by the main script.
if not logger.handlers:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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


class MultiDayBatchSampler(Sampler):
    def __init__(self, data_source, days_per_batch=4, shuffle=False):
        self.data_source = data_source
        self.days_per_batch = days_per_batch
        self.shuffle = shuffle
        self.num_days = len(data_source)
        
    def __iter__(self):
        indices = np.arange(self.num_days)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Group days into batches
        for i in range(0, len(indices), self.days_per_batch):
            yield indices[i:i + self.days_per_batch]
    
    def __len__(self):
        return (self.num_days + self.days_per_batch - 1) // self.days_per_batch


class TemporallyAwareBatchSampler(Sampler):
    def __init__(self, data_source, days_per_batch=4, shuffle=False):
        self.data_source = data_source
        self.days_per_batch = days_per_batch
        self.shuffle = shuffle
        self.num_days = len(data_source)
        
    def __iter__(self):
        # Create consecutive day groups to maintain temporal order
        batch_starts = list(range(0, self.num_days, self.days_per_batch))
        
        if self.shuffle:
            # Shuffle the starting points of batches, not individual days
            np.random.shuffle(batch_starts)
        
        for start_idx in batch_starts:
            end_idx = min(start_idx + self.days_per_batch, self.num_days)
            # Yield consecutive day indices to maintain temporal order within batch
            consecutive_indices = list(range(start_idx, end_idx))
            yield consecutive_indices
    
    def __len__(self):
        return (self.num_days + self.days_per_batch - 1) // self.days_per_batch


class SequenceModel():
    """
    Base class for sequence models.
    Handles training, prediction, saving and loading of the model.
    """
    def __init__(self, n_epochs=100, lr=0.001, GPU=None, seed=None, 
                 train_stop_loss_thred=None, save_path="model/", save_prefix="model",
                 metric="loss", early_stop=20, patience=10, decay_rate=0.9, min_lr=1e-05,
                 max_iters_epoch=None, train_noise=0.0, validation_freq=1, use_amp=True,
                 accumulation_steps=4, max_batch_size=4096):  # Add max batch size for pre-allocation
        """
        Initialize the SequenceModel.

        Parameters:
        - n_epochs (int): Number of training epochs.
        - lr (float): Learning rate.
        - GPU (int or None): GPU ID to use. If None, use CPU.
        - seed (int or None): Random seed for reproducibility.
        - train_stop_loss_thred (float or None): Threshold for early stopping based on training loss.
        - save_path (str): Directory to save the trained model.
        - save_prefix (str): Prefix for the saved model filename.
        - metric (str): Metric to monitor for early stopping (e.g., "loss", "ic").
        - early_stop (int): Number of epochs for early stopping if no improvement.
        - patience (int): Number of epochs to wait before decaying learning rate.
        - decay_rate (float): Factor by which to decay learning rate.
        - min_lr (float): Minimum learning rate.
        - max_iters_epoch (int or None): Maximum number of iterations (batches) per epoch.
        - train_noise (float): Standard deviation of Gaussian noise to add to inputs during training.
        - validation_freq (int): Frequency of validation (in epochs).
        - use_amp (bool): Whether to use mixed precision training.
        - accumulation_steps (int): Number of steps to accumulate gradients before updating optimizer.
        - max_batch_size (int): Maximum batch size for pre-allocation of tensors.
        """
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if GPU is not None and torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.save_path = Path(save_path)
        self.save_prefix = save_prefix
        
        self.metric = metric
        self.early_stop = early_stop
        self.patience = patience
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.max_iters_epoch = max_iters_epoch
        self.train_noise = train_noise
        self.validation_freq = validation_freq  # Only validate every N epochs
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.accumulation_steps = accumulation_steps
        self.max_batch_size = max_batch_size
        self.pre_allocated_tensors = {}  # Cache for reusable tensors

        self.model = None  # This will be set by subclasses (e.g., GRU, LSTM specific models)
        self.optimizer = None
        self.criterion = nn.MSELoss() # Default loss, can be overridden
        self.scheduler = None


        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"SequenceModel initialized. Device: {self.device}, Save Path: {self.save_path}")

    def init_model(self):
        """
        Initializes the optimizer and learning rate scheduler.
        This method should be called after self.model is defined by a subclass.
        """
        if self.model is None:
            raise ValueError("self.model must be defined before calling init_model().")
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min' if 'loss' in self.metric else 'max', # Adjust mode based on metric
            factor=self.decay_rate, patience=self.patience, min_lr=self.min_lr, verbose=True
        )
        logger.info("Optimizer and LR Scheduler initialized.")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        accumulated_loss = 0.0
        
        # Accumulate multiple days worth of data before processing
        accumulated_features = []
        accumulated_labels = []
        current_accumulation = 0

        for i, data in enumerate(data_loader):
            feature_data = data[0]
            label_data = data[1]
            
            # Accumulate data from multiple days
            accumulated_features.append(feature_data)
            accumulated_labels.append(label_data)
            current_accumulation += 1
            
            # Process when we have enough accumulated data or at end
            if current_accumulation >= self.accumulation_steps or i == len(data_loader) - 1:
                # Concatenate all accumulated data
                batch_features = torch.cat(accumulated_features, dim=0).to(self.device, non_blocking=True)
                batch_labels = torch.cat(accumulated_labels, dim=0).to(self.device, non_blocking=True)
                
                # Add noise if specified
                if self.train_noise > 0:
                    noise = torch.randn_like(batch_features) * self.train_noise
                    batch_features = batch_features + noise

                # LEAK-FREE: Replace drop_extreme with robust loss and per-day normalization
                # Apply feature clipping (safe - doesn't use labels)
                processed_features = apply_feature_clipping(batch_features, clip_std=3.0)
                
                # Create day indices for each sample in the batch  
                samples_per_day = processed_features.shape[0] // current_accumulation
                day_indices = torch.arange(current_accumulation, device=self.device, dtype=torch.long).repeat_interleave(samples_per_day)
                
                # Use per-day zscore normalization (safe)
                processed_labels = zscore_per_day(batch_labels.clone(), day_indices)
                
                # No more drop_extreme - use all preprocessed data
                # Split into chunks if too large for GPU memory
                chunk_size = min(len(processed_features), 2048)  # Larger chunks
                total_loss = 0.0
                num_chunks = 0
                
                for chunk_start in range(0, len(processed_features), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(processed_features))
                    chunk_features = processed_features[chunk_start:chunk_end]
                    chunk_labels = processed_labels[chunk_start:chunk_end]
                    
                    if len(chunk_features) == 0:
                        continue
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pred = self.model(chunk_features.float())
                            loss = self.loss_fn(pred, chunk_labels)
                    else:
                        pred = self.model(chunk_features.float())
                        loss = self.loss_fn(pred, chunk_labels)
                    
                    # Scale loss by number of chunks and accumulation steps
                    loss = loss / (current_accumulation * max(1, len(processed_features) // chunk_size))
                    total_loss += loss.item()
                    num_chunks += 1
                    
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Update optimizer
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if num_chunks > 0:
                    losses.append(total_loss)
                
                # Reset accumulation
                accumulated_features = []
                accumulated_labels = []
                current_accumulation = 0

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

    def _init_data_loader(self, data, shuffle=True, drop_last=True, days_per_batch=1):
        if data is None or len(data) == 0:
            raise ValueError("Input data for DataLoader cannot be None or empty.")

        day_sampler = DaySampler(data, shuffle)

        def safe_collate(batch_data_tuple):
            if isinstance(batch_data_tuple, tuple) and len(batch_data_tuple) == 2:
                features, labels = batch_data_tuple
                return (features, labels)
            else:
                raise ValueError(f"Unexpected batch_data_tuple structure: {type(batch_data_tuple)}")

        # Add workers and prefetching for better CPU-GPU overlap
        data_loader = DataLoader(
            data, 
            sampler=day_sampler, 
            batch_size=None, 
            collate_fn=safe_collate,
            num_workers=4,  # Use multiple processes for data loading
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=2,  # Pre-load next batches
            persistent_workers=True  # Keep workers alive between epochs
        )
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        print(f"[SequenceModel] Starting fit method. n_epochs: {self.n_epochs}")
        
        # Monitor GPU usage
        if torch.cuda.is_available():
            print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        if dl_train is None or len(dl_train) == 0 :
            print("[SequenceModel] ERROR: Training data (dl_train) is None or empty in fit method. Cannot proceed.")
            return

        print("[SequenceModel] Initializing training DataLoader...")
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        
        valid_loader = None
        if dl_valid is not None and len(dl_valid) > 0: # Check if dl_valid has data
            print("[SequenceModel] Initializing validation DataLoader...")
            valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=False) # No shuffle for validation

        if train_loader is None : 
             print("[SequenceModel] ERROR: train_loader is None after _init_data_loader. Cannot proceed with training.")
             return

        best_param = None
        best_valid_metric_val = -np.inf if 'ic' in self.metric.lower() else np.inf
        epochs_no_improve = 0

        print("[SequenceModel] Starting training loop...")
        for step in range(self.n_epochs):
            epoch_log_msg_parts = []
            epoch_log_msg_parts.append(f"[SequenceModel] Epoch {step + 1}/{self.n_epochs}")
            
            print(f"{epoch_log_msg_parts[0]} - Starting train_epoch...")
            train_loss = self.train_epoch(train_loader)
            self.fitted = step # Mark as fitted at least one epoch
            epoch_log_msg_parts.append(f"train_loss: {train_loss:.6f}")

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_log_msg_parts.append(f"lr: {current_lr:.6e}")

            # Only validate every validation_freq epochs (default 1 = every epoch)
            if valid_loader and (step + 1) % self.validation_freq == 0:
                print(f"{epoch_log_msg_parts[0]} - Performing validation...")
                # Calculate validation loss
                valid_loss = self.test_epoch(valid_loader) # Ensure test_epoch returns average loss
                epoch_log_msg_parts.append(f"valid_loss: {valid_loss:.6f}")
                
                # Calculate IC metrics for validation
                # self.predict (which will be MASTER.predict if self is MASTER instance)
                # returns a DataFrame: ['date', 'ticker', 'prediction', 'actual_return']
                # The original SequenceModel.predict returned (predictions_series, metrics_dict)
                
                # For validation within SequenceModel.fit, we need ICs.
                # If self.predict is MASTER.predict, it returns a DataFrame.
                # If self.predict is the original SequenceModel.predict, it returns (Series, dict).
                
                # Let's assume for now that if dl_valid is a DailyGroupedTimeSeriesDataset,
                # self.predict will be MASTER.predict and return a DataFrame.
                # We need a consistent way to get metrics.
                
                valid_ic = np.nan
                valid_ric = np.nan

                # Check if self.predict is the one from MASTER that returns a DataFrame
                # This is a bit fragile; ideally, a more robust interface would be used.
                # For now, we try to adapt.
                
                # The self.predict method in the context of MASTER will be MASTER.predict
                # which expects a DailyGroupedTimeSeriesDataset and returns a DataFrame
                validation_output = self.predict(dl_valid) # dl_valid is the Dataset object

                if isinstance(validation_output, pd.DataFrame) and not validation_output.empty:
                    # This is likely the output from MASTER.predict
                    # Calculate daily ICs and then average
                    daily_metrics_val = validation_output.groupby('date').apply(
                        lambda x: pd.Series({
                            'ic': calc_ic(x['prediction'], x['actual_return'])[0],
                            'rank_ic': calc_ic(x['prediction'], x['actual_return'])[1]
                        })
                    ).reset_index()
                    valid_ic = daily_metrics_val['ic'].mean()
                    valid_ric = daily_metrics_val['rank_ic'].mean()
                    logger.info(f"Validation ICs calculated from MASTER.predict DataFrame: IC={valid_ic:.4f}, RankIC={valid_ric:.4f}")
                elif isinstance(validation_output, tuple) and len(validation_output) == 2:
                    # This is likely the output from a base SequenceModel.predict (if it were called)
                    # predictions_series, metrics_dict = validation_output
                    # valid_ic = metrics_dict.get('IC', np.nan)
                    # valid_ric = metrics_dict.get('RIC', np.nan)
                    # logger.info(f"Validation ICs from SequenceModel.predict tuple: IC={valid_ic:.4f}, RankIC={valid_ric:.4f}")
                    # This branch is less likely to be hit if MASTER always overrides predict.
                    # For safety, let's assume MASTER.predict is what we are dealing with.
                    pass # Keep ICs as NaN if we don't get the DataFrame from MASTER.predict
                else:
                    logger.warning(f"Unexpected output type from self.predict during validation: {type(validation_output)}. Cannot calculate ICs.")


                epoch_log_msg_parts.append(f"valid_ic: {valid_ic:.4f}")
                epoch_log_msg_parts.append(f"valid_rank_ic: {valid_ric:.4f}")
                
                # Determine metric for scheduler and early stopping
                metric_for_scheduler = valid_loss # Default to validation loss
                if self.metric == 'ic':
                    metric_for_scheduler = valid_ic
                elif self.metric == 'ric':
                    metric_for_scheduler = valid_ric
                
                self.scheduler.step(metric_for_scheduler)

                # Early stopping logic
                improved = False
                if 'ic' in self.metric.lower(): # Higher is better for IC
                    if metric_for_scheduler > best_valid_metric_val:
                        best_valid_metric_val = metric_for_scheduler
                        improved = True
                else: # Lower is better for loss
                    if metric_for_scheduler < best_valid_metric_val:
                        best_valid_metric_val = metric_for_scheduler
                        improved = True
                
                if improved:
                    epochs_no_improve = 0
                    # Use state_dict() copy instead of deepcopy for efficiency
                    best_param = {k: v.clone() for k, v in self.model.state_dict().items()}
                    
                    # Save asynchronously or less frequently
                    if (step + 1) % 5 == 0 or step == self.n_epochs - 1:  # Save every 5 epochs
                        torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}_best.pkl')
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.early_stop:
                    logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement on validation {self.metric}.")
                    if best_param is not None:
                        logger.info("Loading best model parameters.")
                        self.model.load_state_dict(best_param) # Load best model
                    break
            else:
                # Use training loss for scheduler when not validating
                self.scheduler.step(train_loss)
            
            print(" - ".join(epoch_log_msg_parts))

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                logger.info(f"Training stop threshold based on train_loss met. train_loss: {train_loss:.6f} <= {self.train_stop_loss_thred}")
                if best_param is None : 
                    best_param = copy.deepcopy(self.model.state_dict())
                    torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}_best.pkl')
                    logger.info(f"Model saved to {self.save_path}/{self.save_prefix}_{self.seed}_best.pkl based on train_stop_loss_thred.")
                break

            # Monitor GPU usage during training
            if torch.cuda.is_available() and step % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Epoch {step}: GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        if not (epochs_no_improve >= self.early_stop): # If not early stopped
            if best_param is None and self.n_epochs > 0 : 
                 final_model_state = copy.deepcopy(self.model.state_dict())
                 torch.save(final_model_state, f'{self.save_path}/{self.save_prefix}_{self.seed}_final.pkl')
                 logger.info(f"Training finished. Final model saved to {self.save_path}/{self.save_prefix}_{self.seed}_final.pkl")
            elif best_param is not None and not Path(f'{self.save_path}/{self.save_prefix}_{self.seed}_final.pkl').exists():
                 # If best model was saved due to validation, but training completed all epochs
                 # and we want to ensure the 'best' model is what's considered final if not overwritten
                 logger.info(f"Training finished. Best model (from validation) at {self.save_path}/{self.save_prefix}_{self.seed}_best.pkl is considered the final state for this run if no explicit final save.")


        print("[SequenceModel] Fit method completed.")

    def predict(self, dl_test):
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
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

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return predictions, metrics

    def _get_or_create_tensor(self, key, shape, dtype=torch.float32):
        """Get a pre-allocated tensor or create new one if needed"""
        if key not in self.pre_allocated_tensors or self.pre_allocated_tensors[key].shape[0] < shape[0]:
            self.pre_allocated_tensors[key] = torch.empty(
                shape, dtype=dtype, device=self.device
            )
        return self.pre_allocated_tensors[key][:shape[0]]

    def diagnose_performance(self, dl_train):
        """Diagnostic function to identify performance bottlenecks"""
        print("=== Performance Diagnosis ===")
        
        train_loader = self._init_data_loader(dl_train, shuffle=False, drop_last=False)
        
        # Test data loading speed
        print("Testing data loading speed...")
        start_time = time.time()
        batch_sizes = []
        
        for i, data in enumerate(train_loader):
            if i >= 10:  # Test first 10 batches
                break
            batch_sizes.append(len(data[1]))  # Label batch size
        
        load_time = time.time() - start_time
        avg_batch_size = sum(batch_sizes) / len(batch_sizes)
        
        print(f"Data loading: {load_time:.2f}s for 10 batches")
        print(f"Average batch size: {avg_batch_size:.1f} samples")
        print(f"Batch sizes: {batch_sizes}")
        
        # Test GPU utilization with dummy forward pass
        if hasattr(self, 'model') and self.model is not None:
            print("\nTesting model forward pass...")
            self.model.eval()
            
            # Create dummy batch of different sizes
            for batch_size in [50, 200, 500, 1000, 2000]:
                if hasattr(train_loader.dataset, 'feature_dim') and hasattr(train_loader.dataset, 'sequence_length'):
                    dummy_input = torch.randn(
                        batch_size, 
                        train_loader.dataset.sequence_length, 
                        train_loader.dataset.feature_dim
                    ).to(self.device)
                else:
                    # Assume common dimensions
                    dummy_input = torch.randn(batch_size, 8, 158).to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                forward_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"Batch size {batch_size}: {forward_time*1000:.2f}ms, GPU mem: {memory_used:.3f}GB")
