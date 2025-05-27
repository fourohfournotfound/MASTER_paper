import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

from master import MASTERModel
# Assuming base_model.py contains the definition for SequenceModel
# and other necessary components are correctly imported by master.py
# from base_model import SequenceModel # Already imported in master.py
from base_model import zscore # Importing zscore for label normalization

# Memory and Logging (placeholders for now)
LONG_SHORT_TERM_MEMORY_FILE = "memory_log.json"
LESSONS_LEARNED_FILE = "lessons_learned.md"

def log_to_memory(data):
    # Basic implementation, can be expanded
    print(f"Memory log: {data}")

def log_lesson(lesson):
    # Basic implementation, can be expanded
    print(f"Lesson learned: {lesson}")

def calculate_ic(predictions, actuals):
    """Calculates Spearman Rank Correlation (Information Coefficient)."""
    if len(predictions) < 2 or len(actuals) < 2:
        return 0.0, 0.0 # Not enough data for correlation
    
    # Ensure inputs are 1D arrays
    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()

    # Check for constant arrays which can cause issues with spearmanr
    if np.all(predictions == predictions[0]) or np.all(actuals == actuals[0]):
        log_to_memory({"event": "calculate_ic_constant_array_warning"})
        return 0.0, 1.0 # Return 0 correlation, p-value 1 to indicate issue

    try:
        ic_value, p_value = spearmanr(predictions, actuals)
        if np.isnan(ic_value): # spearmanr can return nan if std dev is zero
            log_to_memory({"event": "calculate_ic_nan_result"})
            return 0.0, 1.0
        return ic_value, p_value
    except Exception as e:
        log_lesson(f"Error calculating IC: {e}")
        return 0.0, 1.0

def load_and_preprocess_data(csv_path, sequence_len=60):
    """
    Loads data from a CSV file, preprocesses it for the MASTER model.

    Args:
        csv_path (str): Path to the input CSV file.
        sequence_len (int): Length of the input sequences for the model.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler_map, target_scaler, d_feat)
               Returns None for all if data loading or processing fails.
    """
    log_to_memory({"event": "load_and_preprocess_data_start", "csv_path": csv_path})

    try:
        df = pd.read_csv(csv_path)
        log_to_memory({"event": "data_loaded", "shape": df.shape})
    except FileNotFoundError:
        log_lesson(f"File not found: {csv_path}. Ensure correct path.")
        print(f"Error: File not found at {csv_path}")
        return (None,) * 9

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    except ValueError as e:
        log_lesson(f"Date conversion error: {e}. Check date format in CSV. Expected YYYY-MM-DD.")
        print(f"Error converting date column: {e}. Please ensure dates are in YYYY-MM-DD format.")
        return (None,) * 9

    df = df.set_index(['ticker', 'date'])
    df = df.sort_index()
    log_to_memory({"event": "multiindex_set_and_sorted"})

    df['returns'] = df.groupby(level='ticker')['closeadj'].pct_change().fillna(0)
    log_to_memory({"event": "returns_calculated"})

    features_to_use = ['open', 'high', 'low', 'close', 'volume', 'closeadj']
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        log_lesson(f"Missing expected features: {missing_features}. Required: {features_to_use}")
        print(f"Error: Missing features in CSV: {missing_features}. Required: {features_to_use}")
        return (None,) * 9
        
    df_features = df[features_to_use].copy()
    df_target = df[['returns']].copy()

    # Handle Missing Values (per ticker) ensuring index integrity
    # Apply ffill/bfill within each group and reconstruct
    filled_features_list = []
    for ticker_name, group in df_features.groupby(level='ticker'):
        filled_group = group.ffill().bfill()
        filled_features_list.append(filled_group)
    
    if filled_features_list:
        df_features = pd.concat(filled_features_list)
    else:
        # Handle case where df_features might be empty or all groups are empty
        df_features = pd.DataFrame(columns=features_to_use) 


    df_features.dropna(inplace=True) 
    # Align target with features after potential NaN drops and ensure index consistency
    # Only keep indices present in both df_features and the original df_target's index levels
    common_index = df_features.index.intersection(df_target.index)
    df_features = df_features.loc[common_index]
    df_target = df_target.loc[common_index]


    log_to_memory({"event": "missing_values_handled", "shape_after_na": df_features.shape})
    if df_features.empty:
        log_lesson("DataFrame became empty after handling missing values.")
        print("Error: DataFrame empty after handling missing values.")
        return (None,) * 9

    log_to_memory({"event": "stationarity_check_skipped_for_now"})

    feature_scaler_map = {}
    scaled_features_list = []
    for ticker, group in df_features.groupby(level='ticker'):
        if not group.empty and not group.isnull().all().all():
            scaler = MinMaxScaler()
            scaled_group_features = scaler.fit_transform(group)
            scaled_features_list.append(pd.DataFrame(scaled_group_features, index=group.index, columns=group.columns))
            feature_scaler_map[ticker] = scaler
        else:
            log_to_memory({"event": "empty_or_all_nan_group_skipped_scaling", "ticker": ticker})

    if not scaled_features_list:
        log_lesson("No data available after attempting to scale features.")
        print("Error: No data to process after feature scaling step.")
        return (None,) * 9
        
    df_scaled_features = pd.concat(scaled_features_list)
    d_feat = df_scaled_features.shape[1]
    log_to_memory({"event": "features_scaled_per_ticker", "d_feat": d_feat})

    temp_target_scaler = StandardScaler()
    df_target_scaled_values = temp_target_scaler.fit_transform(df_target.values.reshape(-1, 1))
    df_target_scaled = pd.DataFrame(df_target_scaled_values, index=df_target.index, columns=df_target.columns)
    log_to_memory({"event": "target_variable_temporarily_scaled"})

    X_list, y_list = [], []
    for ticker, group_data in df_scaled_features.groupby(level='ticker'):
        ticker_features = group_data.values
        ticker_target_scaled = df_target_scaled.loc[group_data.index].values.squeeze()

        if len(ticker_features) >= sequence_len + 1:
            for i in range(len(ticker_features) - sequence_len):
                X_list.append(ticker_features[i:i+sequence_len, :])
                y_list.append(ticker_target_scaled[i+sequence_len]) 
    
    if not X_list:
        log_lesson(f"Not enough data to create sequences of length {sequence_len}.")
        print(f"Error: Insufficient data to create sequences with length {sequence_len}.")
        return (None,) * 9

    X = np.array(X_list)
    y = np.array(y_list)
    log_to_memory({"event": "sequences_created", "X_shape": X.shape, "y_shape": y.shape})

    num_samples = X.shape[0]
    train_size = int(num_samples * 0.7)
    val_size = int(num_samples * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    target_scaler = StandardScaler()
    y_train_reshaped = y_train.reshape(-1, 1)
    target_scaler.fit(y_train_reshaped)
    
    y_train = target_scaler.transform(y_train_reshaped).squeeze() 
    y_val = target_scaler.transform(y_val.reshape(-1, 1)).squeeze()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).squeeze()
    log_to_memory({"event": "target_variable_rescaled_on_train_set_only"})

    log_to_memory({
        "event": "data_split",
        "X_train_shape": X_train.shape, "y_train_shape": y_train.shape,
        "X_val_shape": X_val.shape, "y_val_shape": y_val.shape,
        "X_test_shape": X_test.shape, "y_test_shape": y_test.shape,
    })

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 : 
        log_lesson("Data splitting resulted in empty train or validation sets.")
        print("Error: Train or validation data splits are empty.")
        return (None,) * 9

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler_map, target_scaler, d_feat

def run_training(csv_path, sequence_len=60, d_model=256, t_nhead=4, s_nhead=2, dropout=0.1, beta=1.0, n_epochs=10, lr=1e-4):
    log_to_memory({"event": "run_training_start", "params": locals()})

    processed_data = load_and_preprocess_data(csv_path, sequence_len)
    if processed_data[0] is None:
        log_lesson("Data loading/preprocessing failed. Aborting training.")
        print("Failed to load or preprocess data. Aborting training.")
        return

    X_train, y_train, X_val, y_val, X_test, y_test, _feature_scalers, target_scaler, d_feat = processed_data
    
    if d_feat is None:
        log_lesson("d_feat is None after preprocessing. Aborting training.")
        print("Error: Number of features (d_feat) could not be determined. Aborting training.")
        return

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    X_train_master = torch.cat([X_train_tensor, X_train_tensor], dim=2)
    X_val_master = torch.cat([X_val_tensor, X_val_tensor], dim=2)

    train_dataset = TensorDataset(X_train_master, y_train_tensor)
    val_dataset = TensorDataset(X_val_master, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_primary_features = d_feat 
    master_d_feat_param = num_primary_features
    master_gate_input_start_index_param = num_primary_features
    master_gate_input_end_index_param = num_primary_features * 2
    
    log_to_memory({
        "event": "model_hyperparameters",
        "d_feat_model": master_d_feat_param, "d_model": d_model, "t_nhead": t_nhead, 
        "s_nhead": s_nhead, "dropout": dropout, "beta": beta,
        "gate_input_start_index": master_gate_input_start_index_param,
        "gate_input_end_index": master_gate_input_end_index_param,
        "n_epochs_arg": n_epochs, "lr_arg": lr
    })

    model = MASTERModel(
        d_feat=master_d_feat_param,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=dropout,
        S_dropout_rate=dropout,
        beta=beta,
        gate_input_start_index=master_gate_input_start_index_param,
        gate_input_end_index=master_gate_input_end_index_param,
        n_epochs=n_epochs,
        lr=lr
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_to_memory({"event": "model_initialized_on_device", "device": str(device)})

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)

    log_to_memory({"event": "training_start"})
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 5

    if n_epochs <= 0:
        print(f"Warning: Number of training epochs is {n_epochs}. No training will occur and no epoch metrics will be printed.")
        log_to_memory({"event": "training_skipped_due_to_n_epochs", "n_epochs_value": n_epochs})

    for epoch in range(n_epochs):
        model.model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model.model(batch_X)
            loss = criterion(predictions, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)

        model.model.eval()
        epoch_val_loss = 0
        all_val_preds = []
        all_val_actuals = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_preds = model.model(batch_X_val)
                loss = criterion(val_preds, batch_y_val.squeeze(-1))
                epoch_val_loss += loss.item()
                all_val_preds.append(val_preds.cpu().numpy())
                all_val_actuals.append(batch_y_val.cpu().numpy())
        
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        
        all_val_preds_np = np.concatenate(all_val_preds)
        all_val_actuals_np = np.concatenate(all_val_actuals)
        val_ic, val_p_value = calculate_ic(all_val_preds_np, all_val_actuals_np)

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_epoch_train_loss:.6f}, Val Loss: {avg_epoch_val_loss:.6f}, Val IC: {val_ic:.4f}")
        log_to_memory({
            "event": "epoch_end", "epoch": epoch+1, 
            "train_loss": avg_epoch_train_loss, "val_loss": avg_epoch_val_loss,
            "val_ic": val_ic, "val_ic_p_value": val_p_value
        })

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            epochs_no_improve = 0
            # torch.save(model.model.state_dict(), "best_model_checkpoint.pth")
            # log_to_memory({"event": "best_model_saved", "epoch": epoch+1, "val_loss": best_val_loss})
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            log_to_memory({"event": "early_stopping", "epoch": epoch+1})
            break
            
    log_to_memory({"event": "training_complete"})
    
    # Test phase
    log_to_memory({"event": "test_phase_start"})
    model.model.eval() # Ensure model is in evaluation mode
    
    # Prepare test data - assuming X_test_master is needed if model expects concatenated features
    # Ensure X_test_tensor is correctly formed from preprocessed test data
    if X_test_tensor.shape[2] == num_primary_features: # d_feat from preprocessing
         X_test_master_input = torch.cat([X_test_tensor, X_test_tensor], dim=2).to(device)
    else: # If X_test_tensor was already prepared like X_train_master
         X_test_master_input = X_test_tensor.to(device)

    all_test_preds = []
    all_test_actuals = []
    
    # Create a DataLoader for the test set for batching, if memory is a concern
    # For simplicity, if test set is small enough, can predict in one go
    # Here, let's assume we might need a DataLoader for consistency
    test_dataset = TensorDataset(X_test_master_input, y_test_tensor.to(device)) # y_test_tensor needs to be on device too
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            # batch_X_test is already on device from X_test_master_input
            # batch_y_test is already on device from test_dataset
            test_preds = model.model(batch_X_test)
            all_test_preds.append(test_preds.cpu().numpy())
            all_test_actuals.append(batch_y_test.cpu().numpy())

    all_test_preds_np = np.concatenate(all_test_preds)
    all_test_actuals_np = np.concatenate(all_test_actuals)

    # Inverse transform predictions and actuals for scaled metrics like MSE
    # Predictions are for the scaled target
    # Reshape 1D arrays to 2D (-1, 1) before inverse_transform
    test_preds_rescaled = target_scaler.inverse_transform(all_test_preds_np.reshape(-1, 1))
    test_actuals_rescaled = target_scaler.inverse_transform(all_test_actuals_np.reshape(-1, 1))

    test_mse = np.mean((test_preds_rescaled - test_actuals_rescaled)**2)
    test_ic, test_p_value = calculate_ic(all_test_preds_np, all_test_actuals_np) # IC on scaled or unscaled should be similar for rank

    print(f"Test Results - MSE: {test_mse:.6f}, IC: {test_ic:.4f}")
    log_to_memory({
        "event": "test_phase_end", 
        "test_mse": test_mse, 
        "test_ic": test_ic,
        "test_ic_p_value": test_p_value
    })
    
    print("Training and testing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MASTER model on multi-index stock data.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--sequence_len", type=int, default=60, help="Length of input sequences.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of model internal state.")
    parser.add_argument("--t_nhead", type=int, default=4, help="Number of heads for temporal attention.")
    parser.add_argument("--s_nhead", type=int, default=2, help="Number of heads for spatial attention.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for feature gate.")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    args = parser.parse_args()

    log_to_memory({"event": "main_script_start", "args": vars(args)})
    run_training(
        csv_path=args.csv_path,
        sequence_len=args.sequence_len,
        d_model=args.d_model,
        t_nhead=args.t_nhead,
        s_nhead=args.s_nhead,
        dropout=args.dropout,
        beta=args.beta,
        n_epochs=args.n_epochs,
        lr=args.lr
    )
    log_to_memory({"event": "main_script_end"})

    # TODOs remaining:
    # 1. Stationarity checks (ADF, differencing).
    # 2. Time-series cross-validation.
    # 3. Model saving/loading (best model).
    # 4. Test phase and evaluation metrics (MSE, MAE, IC, RankIC).
    # 5. Feature engineering (technical indicators etc.).
    # 6. Gate feature strategy (currently duplicated primary features).
    # 7. Pytests.
    # 8. Memory logging improvements.
    # 9. SAttention: Critical review of batching for inter-stock relations.
# End of script marker 
