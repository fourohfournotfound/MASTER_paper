import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader

from base_model import SequenceModel, calc_ic

# Setup logger for this module
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(SequenceModel):
    def __init__(self, d_feat, hidden_size, num_layers, dropout, n_epochs, lr, 
                 model_type='GRU', GPU=None, seed=None, train_stop_loss_thred=None,
                 save_path='model/', save_prefix='master_model',
                 metric="loss", early_stop=20, patience=10, decay_rate=0.9, min_lr=1e-05, 
                 max_iters_epoch=None, train_noise=0.0):
        
        # Initialize the parent SequenceModel class
        super().__init__(n_epochs=n_epochs, lr=lr, GPU=GPU, seed=seed,
                         train_stop_loss_thred=train_stop_loss_thred,
                         save_path=save_path, save_prefix=save_prefix,
                         metric=metric, early_stop=early_stop, patience=patience,
                         decay_rate=decay_rate, min_lr=min_lr, max_iters_epoch=max_iters_epoch,
                         train_noise=train_noise)

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type

        # Instantiate the actual PyTorch model (GRU, LSTM, etc.)
        # This model will be assigned to self.model as expected by SequenceModel
        self.model = self._build_model()
        
        # After the model is built and assigned to self.model, initialize optimizer etc.
        self.init_model() # This method is from the parent SequenceModel

    def _build_model(self):
        if self.model_type == 'GRU':
            print(f"Building GRUModel with d_feat={self.d_feat}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}")
            return GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        elif self.model_type == 'LSTM':
            print(f"Building LSTMModel with d_feat={self.d_feat}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}")
            return LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        elif self.model_type == 'ALSTM': # Assuming ALSTM is defined
            print(f"Building ALSTMModel with d_feat={self.d_feat}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}")
            return ALSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from 'GRU', 'LSTM', 'ALSTM'.")

    def fit(self, train_data, valid_data=None):
        """
        Fit the base model (SequenceModel) using the provided training data.
        Optionally, use validation data for early stopping.
        """
        # The SequenceModel (self.model) has its own fit method.
        # We need to prepare the data for it if it's a DailyGroupedTimeSeriesDataset.
        # The SequenceModel.fit expects X_train, y_train directly or a DataLoader
        # that yields (features, labels) not grouped by day.

        # Check type by its name and module to avoid import issues
        train_data_module_name = train_data.__class__.__module__ if hasattr(train_data, '__class__') else None
        is_daily_grouped_dataset_train = (hasattr(train_data, '__class__') and
                                    train_data.__class__.__name__ == 'DailyGroupedTimeSeriesDataset' and
                                    (train_data_module_name == 'main_multi_index' or train_data_module_name == '__main__'))
        
        is_daily_grouped_dataset_valid = False
        if valid_data:
            valid_data_module_name = valid_data.__class__.__module__ if hasattr(valid_data, '__class__') else None
            is_daily_grouped_dataset_valid = (hasattr(valid_data, '__class__') and
                                        valid_data.__class__.__name__ == 'DailyGroupedTimeSeriesDataset' and
                                        (valid_data_module_name == 'main_multi_index' or valid_data_module_name == '__main__'))


        logger.info("Fitting the base sequence model...")
        if is_daily_grouped_dataset_train:
            # SequenceModel.fit expects (X, y) or a specific DataLoader.
            # The current SequenceModel.fit in base_model.py uses _train_model,
            # which expects a DataLoader that yields (features, labels) per batch, not per day.
            # MASTER.__init__ already calls self.model.fit(train_dataset)
            # if n_epochs > 0 and train_dataset is DailyGroupedTimeSeriesDataset.
            # This implies SequenceModel.fit (and _train_model) in base_model.py
            # is expected to handle DailyGroupedTimeSeriesDataset.

            # This explicit self.model.fit(train_data, valid_data) call in MASTER.fit might be
            # for re-training or fine-tuning.
            logger.info("Training with DailyGroupedTimeSeriesDataset. Ensure base_model.SequenceModel.fit can handle this.")
            # Pass valid_data only if it's also the correct type or None
            current_valid_data = valid_data if is_daily_grouped_dataset_valid or valid_data is None else None
            if valid_data is not None and current_valid_data is None:
                logger.warning("Validation data provided to MASTER.fit is not of type DailyGroupedTimeSeriesDataset and will be ignored by super().fit if super().fit also expects this type.")

            super().fit(train_data, current_valid_data) # Call parent's fit method
            logger.info("Base sequence model fitting complete with DailyGroupedTimeSeriesDataset.")

        else:
            # This case assumes train_data is (X_train, y_train) or a compatible DataLoader
            logger.info("Training with standard data (e.g., X_train, y_train, or compatible DataLoader).")
            # Ensure valid_data is not DailyGroupedTimeSeriesDataset if train_data isn't
            current_valid_data = valid_data if not is_daily_grouped_dataset_valid else None
            if valid_data is not None and current_valid_data is None:
                 logger.warning("Validation data provided to MASTER.fit is of type DailyGroupedTimeSeriesDataset but training data is not. Validation data might be incompatible with super().fit expectations and will be ignored.")

            super().fit(train_data, current_valid_data) # Call parent's fit method
            logger.info("Base sequence model fitting complete with standard data.")


    def predict(self, data, dump=False):
        """
        Predict scores using the trained base model.
        data: A DailyGroupedTimeSeriesDataset object or compatible data.
        Returns: A Pandas DataFrame with columns ['date', 'ticker', 'prediction', 'actual_return']
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # Check type by its name and module to avoid import issues
        data_module_name = data.__class__.__module__ if hasattr(data, '__class__') else None
        is_daily_grouped_dataset = (hasattr(data, '__class__') and
                                    data.__class__.__name__ == 'DailyGroupedTimeSeriesDataset' and
                                    (data_module_name == 'main_multi_index' or data_module_name == '__main__'))

        if not is_daily_grouped_dataset:
            # If it's not the special daily grouped dataset.
            logger.error(f"Data object type is '{type(data)}' with module '{data_module_name}'. Expected class 'DailyGroupedTimeSeriesDataset' from module 'main_multi_index' or '__main__'.")
            raise ValueError(
                "This MASTER.predict method is specialized for DailyGroupedTimeSeriesDataset "
                "from 'main_multi_index.py' (or when run as __main__)."
            )

        self.model.eval() # Ensure model is in evaluation mode
        
        all_preds_list = []
        all_labels_list = []
        all_tickers_list = []
        all_dates_list = []

        # DataLoader for DailyGroupedTimeSeriesDataset yields one full day's data per batch
        # batch_size=1 here means 1 day, shuffle=False to maintain temporal order of days.
        data_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

        logger.info(f"Predicting for {len(data.unique_dates)} unique dates...")
        with torch.no_grad():
            for i, (daily_features, daily_labels) in enumerate(data_loader):
                current_date = data.unique_dates[i] # Relies on DataLoader preserving order from dataset
                
                if daily_features.nelement() == 0: # Skip if no data for this day
                    logger.warning(f"No features for date {current_date}, skipping.")
                    continue

                pred = self.model(daily_features.to(self.device)) # (num_stocks_on_day, 1) or (num_stocks_on_day,)
                
                # Ensure pred is squeezed to 1D array of scores
                pred_np = pred.cpu().numpy().squeeze()
                labels_np = daily_labels.numpy().squeeze()

                # Get tickers for the current date
                date_mask = data.multi_index.get_level_values('date') == current_date
                tickers_on_current_date = data.multi_index[date_mask].get_level_values('ticker').tolist()
                
                if len(tickers_on_current_date) == pred_np.shape[0]:
                    all_preds_list.append(pred_np)
                    all_labels_list.append(labels_np)
                    all_tickers_list.extend(tickers_on_current_date)
                    all_dates_list.extend([current_date] * len(tickers_on_current_date))
                else:
                    logger.error(f"Mismatch in number of tickers and predictions for date {current_date}. "
                                 f"Tickers: {len(tickers_on_current_date)}, Preds shape: {pred_np.shape}. Skipping this date.")
                    # This case should ideally not happen if DailyGroupedTimeSeriesDataset is consistent.

        if not all_tickers_list: # Check if any predictions were made
            logger.warning("No predictions were generated. Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'ticker', 'prediction', 'actual_return'])

        # Concatenate lists of arrays only if they are not empty
        flat_predictions = np.concatenate(all_preds_list) if all_preds_list and any(p.size > 0 for p in all_preds_list) else np.array([])
        flat_labels = np.concatenate(all_labels_list) if all_labels_list and any(l.size > 0 for l in all_labels_list) else np.array([])


        # Ensure all resulting arrays have compatible shapes for DataFrame creation
        if not (len(all_dates_list) == len(all_tickers_list) == len(flat_predictions) == len(flat_labels)):
            logger.error(f"Mismatched lengths for DataFrame creation: "
                         f"Dates: {len(all_dates_list)}, Tickers: {len(all_tickers_list)}, "
                         f"Predictions: {len(flat_predictions)}, Labels: {len(flat_labels)}")
            # Attempt to reconcile or return error
            # This indicates a deeper issue in data alignment or processing.
            # For robustness, one might try to align based on the shortest list, but that hides errors.
            # It's better to ensure upstream consistency.
            # If flat_predictions or flat_labels are empty due to all_preds_list/all_labels_list being empty or containing empty arrays
            if flat_predictions.size == 0 or flat_labels.size == 0 and len(all_dates_list) > 0 :
                 logger.warning("Predictions or labels array is empty despite having dates/tickers. This might indicate an issue.")
                 # Fallback for empty predictions/labels but existing dates/tickers
                 # This scenario should be rare if data exists for dates.
                 # Create dataframe with NaNs for missing prediction/labels if lengths match dates/tickers
                 if len(all_dates_list) == len(all_tickers_list):
                     predictions_df = pd.DataFrame({
                        'date': all_dates_list,
                        'ticker': all_tickers_list,
                        'prediction': np.nan if flat_predictions.size == 0 else flat_predictions, # Handle case where flat_predictions might be assigned if only some days had data
                        'actual_return': np.nan if flat_labels.size == 0 else flat_labels
                     })
                     # If flat_predictions/labels were created but are shorter, this will error or broadcast.
                     # Re-check logic for safe construction if some days have no data.
                     # The current append/extend should align, error is for final length check.
                 else: # If dates and tickers also don't align, it's a more fundamental issue.
                    return pd.DataFrame(columns=['date', 'ticker', 'prediction', 'actual_return'])


        predictions_df = pd.DataFrame({
            'date': pd.to_datetime(all_dates_list),
            'ticker': all_tickers_list,
            'prediction': flat_predictions,
            'actual_return': flat_labels
        })
        
        # Sort by date and ticker for consistent output
        predictions_df.sort_values(by=['date', 'ticker'], inplace=True)
        predictions_df.reset_index(drop=True, inplace=True)

        logger.info(f"Prediction complete. Generated {len(predictions_df)} predictions.")
        return predictions_df

    def train_predict(self, train_data, test_data=None, valid_data=None):
        """
        Trains the model using train_data and optionally valid_data,
        and then makes predictions on the test_data.

        Returns:
            predictions_df (pd.DataFrame): DataFrame with predictions on test_data.
                                          Columns: ['date', 'ticker', 'prediction', 'actual_return'].
                                          Returns None if test_data is not provided or empty.
        """
        logger.info("Starting train_predict process in MASTER.")

        if self.model is None: # Should have been initialized in __init__
            logger.error("Base model (self.model) not initialized in MASTER. Cannot train or predict.")
            raise RuntimeError("MASTER's base model not available. Training in __init__ might have failed or was skipped.")

        # --- Training Phase ---
        # self.n_epochs is an attribute of SequenceModel, set during MASTER's super().__init__()
        if self.n_epochs > 0:
            logger.info(f"Calling self.fit to train the model for {self.n_epochs} epochs (from SequenceModel n_epochs)...")
            # MASTER's fit method correctly calls super().fit() which is SequenceModel.fit()
            self.fit(train_data, valid_data) 
            logger.info("Model training completed via self.fit().")
        elif self.fitted < 0 : # self.fitted is -1 if not fitted by SequenceModel.fit
            logger.warning("n_epochs is 0 and model appears unfitted. Predictions will use initial/random weights unless a pre-trained model was loaded.")
        else:
            logger.info("n_epochs is 0, but model appears to be already fitted (or loaded). Skipping training phase.")
        

        # --- Prediction Phase ---
        predictions_df = None
        if test_data is not None:
            logger.info("Making predictions on test data...")
            predictions_df = self.predict(test_data) # This now returns the DataFrame

            if predictions_df is not None and not predictions_df.empty:
                # Calculate IC and Rank IC using the returned DataFrame
                # Group by date to calculate daily IC
                daily_metrics = predictions_df.groupby('date').apply(
                    lambda x: pd.Series({
                        'ic': calc_ic(x['prediction'], x['actual_return'])[0],
                        'rank_ic': calc_ic(x['prediction'], x['actual_return'])[1]
                    })
                ).reset_index()

                mean_ic = daily_metrics['ic'].mean()
                mean_rank_ic = daily_metrics['rank_ic'].mean()
                
                logger.info(f"Test Set Performance: Mean IC: {mean_ic:.4f}, Mean Rank IC: {mean_rank_ic:.4f}")
            else:
                logger.info("No predictions generated for the test set, or test set was empty.")
        else:
            logger.info("No test data provided for prediction.")

        # train_predict should primarily return the predictions for further analysis (like backtesting)
        return predictions_df


class GRUModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0 # Dropout only if num_layers > 1
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_feat)
        out, _ = self.rnn(x) # out shape: (batch_size, seq_len, hidden_size)
        # We want the output from the last time step
        return self.fc(out[:, -1, :]).squeeze() # (batch_size,)


class LSTMModel(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

# Placeholder for ALSTM if you have a specific architecture
class ALSTMModel(LSTMModel): # Example: ALSTM could inherit from LSTM and add attention
    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__(d_feat, hidden_size, num_layers, dropout)
        # Add attention mechanism components if needed
        print("ALSTMModel initialized (currently same as LSTM). Define attention if needed.")


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, hidden_size=self.d_model, num_layers=self.t_nhead, dropout=self.T_dropout_rate,
                           n_epochs=self.n_epochs, lr=self.lr, model_type='GRU', GPU=self.GPU, seed=self.seed,
                           train_stop_loss_thred=self.train_stop_loss_thred, save_path=self.save_path, save_prefix=self.save_prefix)
        super(MASTERModel, self).init_model()
