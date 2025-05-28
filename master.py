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
            for _ in range(nhead): 
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

        # Effective dimension per head, handles non-divisible d_model/nhead for slicing
        head_dim_base = self.d_model // self.nhead
        att_output_per_head = []
        current_dim_offset = 0
        for i in range(self.nhead):
            # Determine actual head_dim for this head
            if i < self.nhead - 1:
                current_head_actual_dim = head_dim_base
            else: # Last head takes the remainder
                current_head_actual_dim = self.d_model - current_dim_offset
            
            # Slice for current head -> qh, kh, vh shape: (T_lookback, N_stocks, current_head_actual_dim)
            q_h = q[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            k_h = k[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            v_h = v[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            current_dim_offset += current_head_actual_dim
            
            # Attention scores for current head: (T_lookback, N_stocks, N_stocks)
            # (T, N, h_dim) @ (T, h_dim, N) -> (T, N, N)
            attn_scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.temperature
            attn_probs_h = torch.softmax(attn_scores_h, dim=-1)

            if self.attn_dropout and self.attn_dropout[i]:
                attn_probs_h = self.attn_dropout[i](attn_probs_h)
            
            # Weighted sum for current head: (T_lookback, N_stocks, current_head_actual_dim)
            # (T, N, N) @ (T, N, h_dim) -> (T, N, h_dim)
            weighted_values_h = torch.matmul(attn_probs_h, v_h)
            att_output_per_head.append(weighted_values_h.transpose(0,1)) # Transpose back to (N_stocks, T_lookback, current_head_actual_dim)

        # Concatenate heads: (N_stocks, T_lookback, d_model)
        att_output_concat = torch.cat(att_output_per_head, dim=-1)
        
        # First residual connection & FFN
        xt = x + att_output_concat # x is (N_stocks, T_lookback, d_model)
        xt_norm2 = self.norm2(xt)
        ffn_output = self.ffn(xt_norm2)
        output = xt + ffn_output # Second residual
        return output


class TAttention(nn.Module):  # Temporal Self-Attention
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
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
            Dropout(p=dropout),
        )
        self.temperature = math.sqrt(d_model / nhead) if nhead > 0 else math.sqrt(d_model)

    def forward(self, x_input):  # x_input shape: (N_stocks, T_lookback, d_model)
        x_norm = self.norm1(x_input)
        q = self.qtrans(x_norm)  # (N, T, D)
        k = self.ktrans(x_norm)  # (N, T, D)
        v = self.vtrans(x_norm)  # (N, T, D)

        batch_size, seq_len, _ = q.shape
        att_output_list = []
        current_dim_offset = 0
        for i in range(self.nhead):
            if i < self.nhead - 1:
                current_head_actual_dim = self.head_dim
            else:
                current_head_actual_dim = self.d_model - current_dim_offset

            q_h = q[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            k_h = k[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            v_h = v[:, :, current_dim_offset : current_dim_offset + current_head_actual_dim]
            current_dim_offset += current_head_actual_dim

            attn_scores_h = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.temperature
            atten_ave_matrixh = torch.softmax(attn_scores_h, dim=-1)

            if self.attn_dropout_modules and self.attn_dropout_modules[i]:
                atten_ave_matrixh = self.attn_dropout_modules[i](atten_ave_matrixh)

            weighted_values_h = torch.matmul(atten_ave_matrixh, v_h)
            att_output_list.append(weighted_values_h)

        att_output_concat = torch.cat(att_output_list, dim=-1)
        xt = x_input + att_output_concat
        xt_norm2 = self.norm2(xt)
        ffn_output = self.ffn(xt_norm2)
        output = xt + ffn_output
        return output


class Gate(nn.Module):
    def __init__(self, d_gate_input_dim):
        super(Gate, self).__init__()
        self.d_gate_input_dim = d_gate_input_dim
        self.d_gate_output_dim = d_gate_input_dim  # Store for scaling
        self.trans = nn.Linear(d_gate_input_dim, d_gate_input_dim)
        self.t = 1.0

    def forward(self, x):
        # Apply the d_output scaling factor as per paper
        return self.d_gate_output_dim * torch.softmax(self.trans(x) / self.t, dim=-1)


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
        self.softplus = nn.Softplus()

    def forward(self, pred, target):
        if pred.dim() > 1: pred = pred.squeeze()
        if target.dim() > 1: target = target.squeeze()

        if pred.shape[0] != target.shape[0]:
            logger.error(f"Shape mismatch in RegRankLoss: pred {pred.shape}, target {target.shape}. Returning 0 loss.")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        if pred.shape[0] < 2: # Need at least 2 stocks for pairwise comparison
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Create pairwise differences
        # pred_ij = pred_i - pred_j
        # target_ij = target_i - target_j
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)  # Shape (N, N)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0) # Shape (N, N)

        # Create a mask for upper triangle (i < j) to consider each pair once
        # and also where target_i != target_j (actual rank difference exists)
        mask = torch.triu(torch.ones_like(pred_diff, dtype=torch.bool), diagonal=1)
        target_rank_diff_mask = (target_diff != 0)
        final_mask = mask & target_rank_diff_mask
        
        if not final_mask.any(): # No valid pairs with rank difference
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Select valid pairs
        valid_pred_diff = pred_diff[final_mask]
        valid_target_diff = target_diff[final_mask]

        # Loss for each pair: softplus(-beta * sign(target_ij) * pred_ij)
        # sign(target_ij) * pred_ij should be positive if ranks agree, negative if disagree.
        # We want to penalize if sign(target_ij) * pred_ij is negative or small positive.
        # loss_terms = self.softplus(-self.beta * torch.sign(valid_target_diff) * valid_pred_diff)
        
        # Alternative formulation often used: log(1 + exp(-sigma * (s_i - s_j))) where target_i > target_j
        # Let's stick to a common pairwise logistic loss formulation:
        # For a pair (i, j), if target_i > target_j, loss_ij = log(1 + exp(-beta * (pred_i - pred_j)))
        # This is equivalent to softplus(-beta * (pred_i - pred_j)) when target_i > target_j.
        # And softplus(-beta * (pred_j - pred_i)) when target_j > target_i.
        # General form: softplus(-beta * sign(target_i - target_j) * (pred_i - pred_j))
        
        loss_terms = self.softplus(-self.beta * torch.sign(valid_target_diff) * valid_pred_diff)
        
        num_pairs = loss_terms.numel()
        if num_pairs == 0: # Should be caught by final_mask.any() but as a safeguard
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        return loss_terms.sum() / num_pairs


# This class implements the architecture from the paper's `MASTER` class
class PaperMASTERArchitecture(nn.Module):
    """
    Exact replica of the MASTER paper's encoder–decoder stack
    ---------------------------------------------------------
    •  A feature–selection Gate driven by market features
    •  Linear → PositionalEncoding → Temporal (causal) Attention
      → Spatial Attention → Final Temporal Attention → Linear decoder
    """

    def __init__(
        self,
        d_feat_input_total: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        gate_input_start_index: int,
        gate_input_end_index: int,
        beta: float,
    ):
        super().__init__()

        # ---------------------------------------------------------------
        # Feature split
        # ---------------------------------------------------------------
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        num_market_features = self.gate_input_end_index - self.gate_input_start_index
        self.d_feat_stock_specific = d_feat_input_total - num_market_features

        if num_market_features <= 0:
            raise ValueError(
                f"gate indices ({gate_input_start_index}, {gate_input_end_index}) "
                "imply zero market features"
            )
        if self.d_feat_stock_specific <= 0:
            raise ValueError("d_feat_stock_specific must be positive")

        # ---------------------------------------------------------------
        # Modules
        # ---------------------------------------------------------------
        self.feature_gate = Gate(
            d_gate_input_dim=num_market_features,
        )

        self.layers = nn.Sequential(
            nn.Linear(self.d_feat_stock_specific, d_model),
            PositionalEncoding(d_model),
            TAttention(
                d_model=d_model,
                nhead=t_nhead,
                dropout=T_dropout_rate,
            ),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            FinalTemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1),
        )

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (N_stocks, T_lookback, d_feat_input_total)

        Returns
        -------
        Tensor
            Shape (N_stocks,) — prediction per stock for the current day
        """
        # 1) Split features ----------------------------------------------------
        if self.gate_input_start_index == 0:                     # market first
            x_stock_specific = x[:, :, self.gate_input_end_index :]
        elif self.gate_input_end_index == x.size(2):             # market last
            x_stock_specific = x[:, :, : self.gate_input_start_index]
        else:                                                    # market mid
            x_stock_specific = torch.cat(
                (
                    x[:, :, : self.gate_input_start_index],
                    x[:, :, self.gate_input_end_index :],
                ),
                dim=2,
            )

        x_market_info = x[:, :, self.gate_input_start_index : self.gate_input_end_index]  # (N, T, market_features)

        # 2) Gate modulation - SAFE VERSION: Apply gate per timestep -------------------
        # Apply gate to each timestep independently to avoid lookahead bias
        gated_stock_features_list = []
        for t in range(x.size(1)):  # For each timestep
            gate_values_t = self.feature_gate(x_market_info[:, t, :])  # (N, market_features)
            gated_features_t = x_stock_specific[:, t, :] * gate_values_t  # (N, stock_features)
            gated_stock_features_list.append(gated_features_t.unsqueeze(1))
        
        gated_stock_features = torch.cat(gated_stock_features_list, dim=1)  # (N, T, stock_features)

        # 3) Encoder–decoder stack --------------------------------------------
        output = self.layers(gated_stock_features).squeeze(-1)        # (N,)

        return output

class MASTERModel(SequenceModel):
    def __init__(self, d_feat, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5, 
                 beta=5.0, gate_input_start_index=0, gate_input_end_index=1, n_epochs=100, lr=0.001, 
                 GPU=None, seed=None, save_path=None, save_prefix="master_model"):

        super().__init__(n_epochs=n_epochs, lr=lr, GPU=GPU, seed=seed,
                         save_path=save_path, save_prefix=save_prefix,
                         metric="loss", 
                         early_stop=20, 
                         patience=10,
                         decay_rate=0.9,
                         min_lr=1e-05,
                         max_iters_epoch=None,
                         train_noise=0.0
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
        
        # Use RegRankLoss instead of MSE as per paper
        self.loss_fn = RegRankLoss(beta=self.beta)
        if self.device.type == 'cuda':
            self.loss_fn = self.loss_fn.to(self.device)
        
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


class MASTERNet(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index):
        super(MASTERNet, self).__init__()
        self.d_feat = d_feat
        self.d_model = d_model
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        # Calculate dimensions
        gate_dim = gate_input_end_index - gate_input_start_index
        factor_dim = gate_input_start_index  # Only pre-market factors as per paper
        
        # Initialize components
        self.gate = Gate(gate_dim)
        self.linear_in = nn.Linear(factor_dim + gate_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.temporal_attention = TAttention(d_model, t_nhead, T_dropout_rate)
        self.spatial_attention = SAttention(d_model, s_nhead, S_dropout_rate)
        self.final_temporal_attention = FinalTemporalAttention(d_model)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (N, T, d_feat)
        N, T, d_feat = x.shape
        
        # Split features as per paper: only keep factors BEFORE market info
        factor_features = x[:, :, :self.gate_input_start_index]  # Only pre-market factors
        market_features = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        
        # Apply Gate per timestep to avoid lookahead bias
        gated_features_list = []
        for t in range(T):
            gate_weights_t = self.gate(market_features[:, t, :])  # (N, gate_dim)
            if factor_features.shape[2] > 0:
                # Apply gate to factor features at timestep t
                gated_factors_t = factor_features[:, t, :] * gate_weights_t
                combined_t = torch.cat([gated_factors_t, market_features[:, t, :]], dim=1)
            else:
                combined_t = market_features[:, t, :] * gate_weights_t
            gated_features_list.append(combined_t.unsqueeze(1))
        
        combined_features = torch.cat(gated_features_list, dim=1)  # (N, T, combined_dim)
        
        # Apply the attention layers
        x = self.linear_in(combined_features)  # (N, T, d_model)
        x = self.pos_encoding(x)
        x = self.temporal_attention(x)
        x = self.spatial_attention(x)
        x = self.final_temporal_attention(x)  # (N, d_model)
        x = self.linear_out(x)  # (N, 1)
        
        return x.squeeze(-1)  # (N,)
