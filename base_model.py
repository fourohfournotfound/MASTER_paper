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
        day_sampler = DaySampler(data, shuffle)
        # DataLoader with a sampler will pass each sampled index to dataset.__getitem__
        # and then collate the results. If batch_size is None (default when sampler is given),
        # it processes one sample at a time. The default collate_fn will wrap the result in a list.
        # So, if dataset.__getitem__ returns (tensor_A, tensor_B),
        # the loop `for data_batch in data_loader:` will give data_batch = [(tensor_A, tensor_B)].
        # We need to extract the tuple.
\
        def custom_collate(batch_list):
            # batch_list is expected to be [features_for_day_tensor, labels_for_day_tensor]
            # when dataset.__getitem__ returns (features_for_day_tensor, labels_for_day_tensor)
            # and batch_size=None with a sampler.
            # print(f"Custom Collate: batch_list length: {len(batch_list)}") # Expect 2
            if len(batch_list) == 2:
                features, labels = batch_list[0], batch_list[1]
                f_shape = features.shape if hasattr(features, 'shape') else type(features)
                l_shape = labels.shape if hasattr(labels, 'shape') else type(labels)
                # print(f"Custom Collate: features shape: {f_shape}, labels shape: {l_shape}")
                return (features, labels) # Return the tuple (feature_tensor, label_tensor)
            else:
                # print(f"Custom Collate: batch_list does not have 2 elements as expected. Length: {len(batch_list)}")
                # This case indicates an issue with DataLoader behavior or dataset output
                raise ValueError(f"Collate function expected batch_list of length 2, got {len(batch_list)}")

        data_loader = DataLoader(data, sampler=day_sampler, batch_size=None, collate_fn=custom_collate)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR']))
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss))

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                break

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
