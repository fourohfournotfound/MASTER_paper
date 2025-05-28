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
import torch.optim as optim

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


# Official SAttention structure (processes N, T, D_model)
class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead) if nhead > 0 else math.sqrt(d_model)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        if dropout > 0 and nhead > 0:
            for _ in range(nhead): # Changed i to _
                attn_dropout_layer.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(attn_dropout_layer)
        else:
            self.attn_dropout = None
        
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x): # x shape: (N_stocks, T_lookback, d_model)
        x_norm = self.norm1(x)
        
        # q, k, v shape: (T_lookback, N_stocks, d_model) after transpose
        q = self.qtrans(x_norm).transpose(0, 1)
        k = self.ktrans(x_norm).transpose(0, 1)
        v = self.vtrans(x_norm).transpose(0, 1)

        head_dim = self.d_model // self.nhead
        att_output_per_head = []
        for i in range(self.nhead):
            # Slice for current head -> qh, kh, vh shape: (T_lookback, N_stocks, head_dim)
            q_h = q[:, :, i * head_dim: (i + 1) * head_dim]
            k_h = k[:, :, i * head_dim: (i + 1) * head_dim]
            v_h = v[:, :, i * head_dim: (i + 1) * head_dim]
            
            # Attention scores for current head: (T_lookback, N_stocks, N_stocks)
            # (T, N, h_dim) @ (T, h_dim, N) -> (T, N, N)
            attn_scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.temperature
            attn_probs_h = torch.softmax(attn_scores_h, dim=-1)

            if self.attn_dropout and self.attn_dropout[i]:
                attn_probs_h = self.attn_dropout[i](attn_probs_h)
            
            # Weighted sum for current head: (T_lookback, N_stocks, head_dim)
            # (T, N, N) @ (T, N, h_dim) -> (T, N, h_dim)
            weighted_values_h = torch.matmul(attn_probs_h, v_h)
            att_output_per_head.append(weighted_values_h.transpose(0,1)) # Transpose back to (N_stocks, T_lookback, head_dim)

        # Concatenate heads: (N_stocks, T_lookback, d_model)
        att_output_concat = torch.cat(att_output_per_head, dim=-1)
        
        # First residual connection & FFN
        xt = x + att_output_concat # x is (N_stocks, T_lookback, d_model)
        xt_norm2 = self.norm2(xt)
        ffn_output = self.ffn(xt_norm2)
        output = xt + ffn_output # Second residual
        return output


class TAttention(nn.Module): # Assuming this is the paper's causal TAttention
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout_modules = []
        if dropout > 0 and nhead > 0:
            for _ in range(nhead):
                self.attn_dropout_modules.append(Dropout(p=dropout))
            self.attn_dropout_modules = nn.ModuleList(self.attn_dropout_modules)
        else:
            self.attn_dropout_modules = None

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )
        self.temperature = math.sqrt(d_model / nhead) if nhead > 0 else math.sqrt(d_model)

    def forward(self, x_input): # x_input shape: (N_stocks, T_lookback, d_model)
        x_norm = self.norm1(x_input)
        q = self.qtrans(x_norm) # (N, T, D)
        k = self.ktrans(x_norm) # (N, T, D)
        v = self.vtrans(x_norm) # (N, T, D)

        batch_size, seq_len, _ = q.shape 
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device, dtype=q.dtype), diagonal=1)
        
        head_dim = self.d_model // self.nhead
        att_output_list = []
        for i in range(self.nhead):
            q_h = q[:, :, i * head_dim: (i+1) * head_dim] # (N, T, h_dim)
            k_h = k[:, :, i * head_dim: (i+1) * head_dim] # (N, T, h_dim)
            v_h = v[:, :, i * head_dim: (i+1) * head_dim] # (N, T, h_dim)
            
            attn_scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.temperature # (N, T, T)
            attn_scores_h = attn_scores_h + causal_mask # Apply causal mask
            atten_ave_matrixh = torch.softmax(attn_scores_h, dim=-1)

            if self.attn_dropout_modules and self.attn_dropout_modules[i]:
                atten_ave_matrixh = self.attn_dropout_modules[i](atten_ave_matrixh)
            
            weighted_values_h = torch.matmul(atten_ave_matrixh, v_h) # (N, T, h_dim)
            att_output_list.append(weighted_values_h)
        
        att_output_concat = torch.cat(att_output_list, dim=-1) # (N, T, D)
        xt = x_input + att_output_concat
        xt_norm2 = self.norm2(xt)
        ffn_output = self.ffn(xt_norm2)
        output = xt + ffn_output
        return output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0): # d_output here is d_feat_stock_specific
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_gate_output_dim = d_output # This is d_feat_stock_specific
        self.t = beta # Temperature for softmax

    def forward(self, gate_input_market_features): # gate_input shape (N_stocks, d_market_features)
        output = self.trans(gate_input_market_features) # (N_stocks, d_feat_stock_specific)
        output = torch.softmax(output / self.t, dim=-1)
        # As per paper's MASTER.forward: src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        # This means the Gate output should directly be the multiplicative factors.
        # The d_output * output scaling is not in the paper's MASTER class forward pass for the gate.
        # The scaling is by d_feat_stock_specific, which is implicitly handled if trans outputs that dim.
        return output # Output shape (N_stocks, d_feat_stock_specific)


# This is the paper's "final" TemporalAttention that aggregates T dimension
class FinalTemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False) # Paper uses d_model here

    def forward(self, z): # z shape: (N_stocks, T_lookback, d_model)
        h = self.trans(z) # (N_stocks, T_lookback, d_model)
        query = h[:, -1, :].unsqueeze(-1) # Query from last time step: (N_stocks, d_model, 1)
        
        # lam shape: (N_stocks, T_lookback)
        lam = torch.matmul(h, query).squeeze(-1) # (N, T, D) @ (N, D, 1) -> (N, T, 1) -> (N,T)
        lam = torch.softmax(lam, dim=1).unsqueeze(1) # (N_stocks, 1, T_lookback) - attention weights over time
        
        # output shape: (N_stocks, d_model)
        output = torch.matmul(lam, z).squeeze(1) # (N,1,T) @ (N,T,D) -> (N,1,D) -> (N,D)
        return output


class RegRankLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        # pred and target are expected to be 1D tensors of shape (N_stocks_in_day,)
        if pred.dim() > 1: pred = pred.squeeze()
        if target.dim() > 1: target = target.squeeze()
        
        # Ensure they have the same number of elements
        if pred.shape[0] != target.shape[0] or pred.shape[0] < 2 : # Need at least 2 for pairwise comparison
            # Return a simple MSE or zero loss if not enough elements for ranking
            # This might happen if a day has only 1 stock after filtering.
            # Or, if shapes mismatch, it's an error.
            # For simplicity, using MSE if not enough elements for ranking.
            # Consider if this day should be skipped or handled differently during training.
            if pred.shape[0] != target.shape[0]:
                # This is more serious, indicates a bug or data issue
                logger.error(f"Shape mismatch in RegRankLoss: pred {pred.shape}, target {target.shape}")
                return torch.tensor(0.0, device=pred.device, requires_grad=True) # Or raise error

            # If only 0 or 1 stock, ranking loss is not well-defined.
            # Return 0 loss or handle as per paper's implementation for such cases.
            # For now, if less than 2 stocks, let's return 0.
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


        # Create pairwise differences for predictions and targets
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0) # Shape (N, N)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0) # Shape (N, N)

        # Mask for valid pairs (upper triangle, i < j)
        mask = torch.triu(torch.ones_like(pred_diff, dtype=torch.bool), diagonal=1)

        # Select only the valid pairs
        pred_diff_pairs = pred_diff[mask]
        target_diff_pairs = target_diff[mask]

        # Loss calculation: sum of logistic losses for misordered pairs
        # A pair is misordered if target_diff > 0 (target_i > target_j) but pred_diff < 0 (pred_i < pred_j)
        # or target_diff < 0 (target_i < target_j) but pred_diff > 0 (pred_i > pred_j)
        # This is equivalent to - sign(target_diff) * pred_diff
        
        # We want to penalize when sign(pred_diff) != sign(target_diff)
        # Loss = log(1 + exp(-beta * pred_diff_pairs * sign(target_diff_pairs)))
        # Handle target_diff_pairs == 0 case (no preference) - these pairs should have 0 loss contribution.
        # For numerical stability and to avoid issues with sign(0), let's filter them.
        
        valid_target_mask = target_diff_pairs != 0
        if not valid_target_mask.any(): # No pairs with rank difference
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_diff_valid = pred_diff_pairs[valid_target_mask]
        target_diff_valid = target_diff_pairs[valid_target_mask]
        
        # loss_pairs = torch.log(1 + torch.exp(-self.beta * pred_diff_valid * torch.sign(target_diff_valid)))
        # Simpler form if target_diff_valid only contains ranking (e.g. 1 if i > j, -1 if i < j)
        # More general: -log(sigmoid(beta * pred_diff * sign(target_diff))) is common
        # Using the formulation: sum over (i,j) where target_i > target_j of log(1 + exp(-beta * (pred_i - pred_j)))
        
        # Let's use the pairwise logistic loss formulation:
        # For each pair (i, j), if target_i > target_j, we want pred_i > pred_j. Loss is log(1 + exp(-beta * (pred_i - pred_j)))
        # If target_i < target_j, we want pred_i < pred_j. Loss is log(1 + exp(-beta * (pred_j - pred_i)))
        # This can be simplified: sum_{i,j s.t. target_i > target_j} log(1 + exp(-beta * (pred_i - pred_j)))
        
        loss = 0.0
        num_pairs = 0
        
        # Iterate through unique pairs (i, j) where i < j
        for i in range(pred.shape[0]):
            for j in range(i + 1, pred.shape[0]):
                p_i, p_j = pred[i], pred[j]
                t_i, t_j = target[i], target[j]
                
                if t_i > t_j: # If stock i should rank higher than stock j
                    loss += torch.log(1 + torch.exp(-self.beta * (p_i - p_j)))
                    num_pairs += 1
                elif t_j > t_i: # If stock j should rank higher than stock i
                    loss += torch.log(1 + torch.exp(-self.beta * (p_j - p_i)))
                    num_pairs += 1
        
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=pred.device, requires_grad=True)


# This class implements the architecture from the paper's `MASTER` class
class PaperMASTERArchitecture(nn.Module):
    def __init__(self, d_feat_input_total, d_model, t_nhead, s_nhead, 
                 T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta):
        super().__init__()
        
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        num_market_features = self.gate_input_end_index - self.gate_input_start_index
        # d_feat_stock_specific is the number of features that are NOT the market feature(s)
        self.d_feat_stock_specific = d_feat_input_total - num_market_features

        if not (0 <= self.gate_input_start_index < d_feat_input_total and 
                self.gate_input_start_index < self.gate_input_end_index <= d_feat_input_total):
            raise ValueError(f"Gate indices [{self.gate_input_start_index},{self.gate_input_end_index}) "
                             f"are out of bounds for d_feat_input_total={d_feat_input_total}")
        if num_market_features <= 0:
             raise ValueError(f"Number of market features (determined by gate indices) must be positive, got {num_market_features}")
        if self.d_feat_stock_specific <=0:
            raise ValueError(f"Number of stock specific features (d_feat_input_total - num_market_features) is {self.d_feat_stock_specific}, must be positive.")

        # Gate takes market features as input and outputs a modulation vector of size d_feat_stock_specific
        self.feature_gate = Gate(num_market_features, self.d_feat_stock_specific, beta=beta)

        self.layers = nn.Sequential(
            nn.Linear(self.d_feat_stock_specific, d_model), # Operates on stock-specific features
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate), 
            FinalTemporalAttention(d_model=d_model), 
            nn.Linear(d_model, 1)
        )
        logger.info(f"PaperMASTERArchitecture: Input d_feat_total={d_feat_input_total}, "
                    f"d_feat_stock_specific={self.d_feat_stock_specific}, "
                    f"num_market_features_for_gate={num_market_features}, d_model={d_model}")

    def forward(self, x): # x shape: (N_stocks, T_lookback, d_feat_input_total)
        
        # Correctly split stock-specific features and the single market feature
        # Assuming the market feature is at gate_input_start_index (and is the only one)
        
        # Create a mask for stock-specific features
        all_indices = torch.arange(x.size(2), device=x.device)
        market_feature_indices = torch.arange(self.gate_input_start_index, self.gate_input_end_index, device=x.device)
        
        # Identify stock-specific feature indices by excluding market feature indices
        # This is more robust if market features aren't strictly at the end or start.
        # However, load_and_prepare_data currently appends the market feature as the LAST column.
        # So, gate_input_start_index points to it, and gate_input_end_index is gate_input_start_index + 1.
        
        # Stock specific features are all columns EXCEPT the market feature column(s)
        if self.gate_input_start_index == 0: # Market feature is the first
            x_stock_specific = x[:, :, self.gate_input_end_index:]
        elif self.gate_input_end_index == x.size(2): # Market feature is the last (our current case)
            x_stock_specific = x[:, :, :self.gate_input_start_index]
        else: # Market feature is in the middle
            x_stock_specific_part1 = x[:, :, :self.gate_input_start_index]
            x_stock_specific_part2 = x[:, :, self.gate_input_end_index:]
            x_stock_specific = torch.cat((x_stock_specific_part1, x_stock_specific_part2), dim=2)
            
        # Market features from the last time step for the gate
        # x_market_info shape: (N_stocks, num_market_features)
        x_market_info = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]

        if x_stock_specific.size(2) != self.d_feat_stock_specific:
            raise RuntimeError(f"Mismatch in expected stock_specific features. Expected {self.d_feat_stock_specific}, got {x_stock_specific.size(2)}")
        if x_market_info.size(1) != (self.gate_input_end_index - self.gate_input_start_index):
             raise RuntimeError(f"Mismatch in market_info features. Expected {self.gate_input_end_index - self.gate_input_start_index}, got {x_market_info.size(1)}")


        # Apply gate to stock-specific features
        # gate_values shape: (N_stocks, d_feat_stock_specific)
        gate_values = self.feature_gate(x_market_info)
        
        # Multiply gate_values with x_stock_specific, broadcasting gate_values across T_lookback
        # gated_stock_features shape: (N_stocks, T_lookback, d_feat_stock_specific)
        gated_stock_features = x_stock_specific * gate_values.unsqueeze(1)
        
        output = self.layers(gated_stock_features).squeeze(-1) 
        return output


class MASTERModel(SequenceModel):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 beta, gate_input_start_index, gate_input_end_index, 
                 n_epochs, lr, GPU, seed, save_path, save_prefix, 
                 **kwargs):

        super().__init__(n_epochs=n_epochs, lr=lr, GPU=GPU, seed=seed,
                         save_path=save_path, save_prefix=save_prefix,
                         metric=kwargs.get('metric', "loss"), 
                         early_stop=kwargs.get('early_stop', 20), 
                         patience=kwargs.get('patience', 10),
                         decay_rate=kwargs.get('decay_rate', 0.9),
                         min_lr=kwargs.get('min_lr', 1e-05),
                         max_iters_epoch=kwargs.get('max_iters_epoch', None),
                         train_noise=kwargs.get('train_noise', 0.0)
                         )
        # d_feat passed here is d_feat_input_total from main_multi_index.py
        self.d_feat_input_total = d_feat 
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.beta = beta 
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        self.model = PaperMASTERArchitecture(
            d_feat_input_total=self.d_feat_input_total, # Pass the total feature dim
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index, # Index in the total feature dim
            gate_input_end_index=self.gate_input_end_index,     # Index in the total feature dim
            beta=self.beta 
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Defaulting to MSELoss as per paper's SequenceModel.loss_fn
        self.loss_fn = nn.MSELoss().to(self.device)
        logger.info("Using MSELoss for MASTERModel training.")
        # Optional: if beta is explicitly for RegRankLoss and desired:
        # if self.beta is not None and self.beta > 0 and kwargs.get('use_regrank_loss', False):
        #     self.loss_fn = RegRankLoss(beta=self.beta).to(self.device)
        #     logger.info(f"Using RegRankLoss with beta={self.beta}")
        # else:
        #     self.loss_fn = nn.MSELoss().to(self.device)
        #     logger.info("Using MSELoss.")
        
        logger.info("MASTERModel (Paper Architecture Replica) initialized.")


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


class ALSTMModel(LSTMModel):
    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__(d_feat, hidden_size, num_layers, dropout)
        # Add attention mechanism components if needed
        print("ALSTMModel initialized (currently same as LSTM). Define attention if needed.")
