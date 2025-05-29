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
from pathlib import Path

from base_model import calc_ic, zscore, drop_extreme  # Only import what we need

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

        # OPTIMIZED: Use single dropout instead of per-head dropout
        self.attn_dropout = nn.Dropout(p=dropout) if dropout > 0 and nhead > 0 else None
        
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

            # OPTIMIZED: Use single dropout for all heads
            if self.attn_dropout:
                attn_probs_h = self.attn_dropout(attn_probs_h)
            
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

        # OPTIMIZED: Use single dropout instead of per-head dropout
        self.attn_dropout = nn.Dropout(p=dropout) if dropout > 0 and nhead > 0 else None

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
        
        # FIXED: Create causal mask to prevent lookahead bias
        # Each timestep can only attend to past and current timesteps
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x_input.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
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
            # FIXED: Apply causal mask to prevent attending to future timesteps
            attn_scores_h = attn_scores_h + causal_mask.unsqueeze(0)  # Broadcast across batch dimension
            atten_ave_matrixh = torch.softmax(attn_scores_h, dim=-1)

            # OPTIMIZED: Use single dropout for all heads
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout(atten_ave_matrixh)

            weighted_values_h = torch.matmul(atten_ave_matrixh, v_h)
            att_output_list.append(weighted_values_h)

        att_output_concat = torch.cat(att_output_list, dim=-1)
        xt = x_input + att_output_concat
        xt_norm2 = self.norm2(xt)
        ffn_output = self.ffn(xt_norm2)
        output = xt + ffn_output
        return output


class Gate(nn.Module):
    """
    CORRECTED Gate implementation matching MASTER paper Equation 1.
    
    Paper equation: α(m_τ) = F · softmax_β(W_α m_τ + b_α)
    where:
    - m_τ is market info (market_dim)
    - F is number of stock features (stock_feature_dim) 
    - β is temperature parameter
    - W_α maps market_dim -> stock_feature_dim
    """
    def __init__(self, market_dim, stock_feature_dim, beta=1.0):
        super(Gate, self).__init__()
        self.market_dim = market_dim
        self.stock_feature_dim = stock_feature_dim
        self.beta = beta
        
        # Linear transformation: market_dim -> stock_feature_dim
        self.trans = nn.Linear(market_dim, stock_feature_dim)

    def forward(self, market_info):
        """
        Args:
            market_info: (N, market_dim) - market features
            
        Returns:
            gate_weights: (N, stock_feature_dim) - gating coefficients
        """
        # Apply linear transformation + bias
        logits = self.trans(market_info)  # (N, stock_feature_dim)
        
        # Apply temperature-scaled softmax
        probabilities = torch.softmax(logits / self.beta, dim=-1)  # (N, stock_feature_dim)
        
        # Scale by F (number of features) as in paper equation
        gate_weights = self.stock_feature_dim * probabilities  # (N, stock_feature_dim)
        
        return gate_weights


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

        # Handle scalar tensors (0-dimensional)
        if pred.dim() == 0 or target.dim() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
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


class ListFoldLoss(nn.Module):
    """
    ListFold Loss for Long-Short Portfolio Strategies
    
    Based on: "Constructing long-short stock portfolio with a new listwise learn-to-rank algorithm"
    by Zhang et al. (2020)
    
    This loss is specifically designed for long-short strategies where both top and bottom
    rankings matter equally, making it ideal for stock portfolio construction.
    
    Two variants:
    - ListFold-exp: Uses exponential transformation (better theoretical properties)
    - ListFold-sgm: Uses sigmoid transformation (consistent with binary classification)
    """
    
    def __init__(self, transformation='exponential', beta=1.0):
        super().__init__()
        self.transformation = transformation
        self.beta = beta
        
        if transformation == 'exponential':
            self.psi = lambda x: torch.exp(self.beta * x)
        elif transformation == 'sigmoid':
            self.psi = lambda x: torch.sigmoid(self.beta * x)
        else:
            raise ValueError("transformation must be 'exponential' or 'sigmoid'")
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted scores for stocks, shape (N,) or (N, 1)
            target: Target scores/labels for ranking, shape (N,) or (N, 1)
        
        Returns:
            ListFold loss value
        """
        if pred.dim() > 1: 
            pred = pred.squeeze()
        if target.dim() > 1: 
            target = target.squeeze()
            
        # Handle edge cases
        if pred.dim() == 0 or target.dim() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        if pred.shape[0] != target.shape[0]:
            logger.error(f"Shape mismatch in ListFoldLoss: pred {pred.shape}, target {target.shape}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        n_stocks = pred.shape[0]
        
        # Need even number of stocks for pairing, if odd, ignore the middle one
        if n_stocks < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Sort stocks by target to get ground truth ranking
        sorted_indices = torch.argsort(target, descending=True)
        sorted_pred = pred[sorted_indices]
        
        # For odd number of stocks, we'll use n_pairs = (n_stocks // 2)
        n_pairs = n_stocks // 2
        
        # If we have odd number of stocks, we'll only use the even number of top stocks
        if n_stocks % 2 == 1:
            sorted_pred = sorted_pred[:-1]  # Remove the middle stock
            n_stocks = n_stocks - 1
        
        loss_terms = []
        
        # ListFold loss: pair i-th stock with (2n+1-i)-th stock
        for i in range(n_pairs):
            # Indices for top and bottom stocks in the sorted order
            top_idx = i  # i-th best stock
            bottom_idx = n_stocks - 1 - i  # (2n+1-i)-th stock (from bottom)
            
            # Scores of the paired stocks
            f_top = sorted_pred[top_idx]
            f_bottom = sorted_pred[bottom_idx]
            
            # Numerator: ψ(f_i - f_{2n+1-i})
            numerator = self.psi(f_top - f_bottom)
            
            # Denominator: sum of ψ(f_u - f_v) for all valid pairs at this step
            # This includes all remaining unpaired stocks
            remaining_start = i
            remaining_end = n_stocks - i
            
            if remaining_end <= remaining_start + 1:
                # Not enough stocks left for pairing
                break
                
            denominator = torch.tensor(0.0, device=pred.device)
            
            # Sum over all pairs (u,v) where remaining_start <= u != v <= remaining_end-1
            for u in range(remaining_start, remaining_end):
                for v in range(remaining_start, remaining_end):
                    if u != v:
                        denominator += self.psi(sorted_pred[u] - sorted_pred[v])
            
            # Avoid division by zero
            if denominator <= 0:
                denominator = torch.tensor(1e-8, device=pred.device)
            
            # Add log term for this pair
            loss_term = torch.log(numerator) - torch.log(denominator)
            loss_terms.append(loss_term)
        
        if not loss_terms:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Return negative log-likelihood (we want to minimize the loss)
        total_loss = -torch.stack(loss_terms).sum()
        
        return total_loss


class ListFoldLossOptimized(nn.Module):
    """
    Optimized version of ListFold Loss for better computational efficiency.
    
    This version reduces the computational complexity of the denominator calculation
    while maintaining the core ListFold algorithm properties.
    """
    
    def __init__(self, transformation='exponential', beta=1.0):
        super().__init__()
        self.transformation = transformation
        self.beta = beta
        
        if transformation == 'exponential':
            self.psi = lambda x: torch.exp(self.beta * x)
        elif transformation == 'sigmoid':
            self.psi = lambda x: torch.sigmoid(self.beta * x)
        else:
            raise ValueError("transformation must be 'exponential' or 'sigmoid'")
    
    def forward(self, pred, target):
        """Optimized ListFold loss computation"""
        if pred.dim() > 1: pred = pred.squeeze()
        if target.dim() > 1: target = target.squeeze()
            
        if pred.dim() == 0 or target.dim() == 0 or pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        n_stocks = pred.shape[0]
        
        # Sort by target (ground truth ranking)
        sorted_indices = torch.argsort(target, descending=True)
        sorted_pred = pred[sorted_indices]
        
        # Use even number of stocks
        if n_stocks % 2 == 1:
            sorted_pred = sorted_pred[:-1]
            n_stocks = n_stocks - 1
        
        n_pairs = n_stocks // 2
        
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for i in range(n_pairs):
            top_idx = i
            bottom_idx = n_stocks - 1 - i
            
            # Score difference for this pair
            score_diff = sorted_pred[top_idx] - sorted_pred[bottom_idx]
            
            # Numerator: ψ(score_diff)
            numerator = self.psi(score_diff)
            
            # Simplified denominator: sum over remaining stocks
            remaining_scores = sorted_pred[i:n_stocks-i]
            
            # Pairwise differences for remaining stocks
            if len(remaining_scores) >= 2:
                score_matrix = remaining_scores.unsqueeze(0) - remaining_scores.unsqueeze(1)
                # Remove diagonal elements (self-differences)
                mask = ~torch.eye(len(remaining_scores), dtype=torch.bool, device=pred.device)
                valid_diffs = score_matrix[mask]
                denominator = torch.sum(self.psi(valid_diffs))
            else:
                denominator = torch.tensor(1e-8, device=pred.device)
            
            # Add log probability for this pair
            if numerator > 0 and denominator > 0:
                total_loss -= (torch.log(numerator) - torch.log(denominator))
        
        return total_loss


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
            raise ValueError(f"d_feat_stock_specific ({self.d_feat_stock_specific}) must be positive. Check feature splitting and gate indices. Total features: {d_feat_input_total}, Market features: {num_market_features}")

        # ---------------------------------------------------------------
        # Modules - OPTIMIZED: Create modules directly without Sequential wrapper overhead
        # ---------------------------------------------------------------
        self.feature_gate = Gate(
            market_dim=num_market_features,
            stock_feature_dim=self.d_feat_stock_specific,
            beta=beta
        )

        # OPTIMIZED: Create individual modules instead of Sequential for faster init
        self.input_linear = nn.Linear(self.d_feat_stock_specific, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.temporal_attention = TAttention(
            d_model=d_model,
            nhead=t_nhead,
            dropout=T_dropout_rate,
        )
        self.spatial_attention = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.final_temporal_attention = FinalTemporalAttention(d_model=d_model)
        self.output_linear = nn.Linear(d_model, 1)

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N_stocks, T_lookback, d_feat_input_total)
                              where N_stocks is batch size (number of stocks for this daily batch),
                              T_lookback is the sequence length,
                              d_feat_input_total is the total number of raw input features.

        Returns:
            torch.Tensor: Output tensor of shape (N_stocks, 1) representing predicted scores/ranks.
        """
        # ---------------------------------------------------------------
        # Split input into stock-specific features and market info
        # ---------------------------------------------------------------
        # x_stock_specific: (N, T, d_feat_stock_specific)
        # x_market_info:    (N, T, num_market_features)
        
        # Create masks for slicing
        stock_specific_indices = list(range(self.gate_input_start_index)) + \
                                 list(range(self.gate_input_end_index, x.shape[2]))
        market_info_indices = list(range(self.gate_input_start_index, self.gate_input_end_index))

        x_stock_specific = x[:, :, stock_specific_indices]
        x_market_info = x[:, :, market_info_indices]

        if x_market_info.shape[2] != (self.gate_input_end_index - self.gate_input_start_index):
             raise ValueError(f"Market info features dimension mismatch. Expected {self.gate_input_end_index - self.gate_input_start_index}, got {x_market_info.shape[2]}")
        if x_stock_specific.shape[2] != self.d_feat_stock_specific:
            raise ValueError(f"Stock specific features dimension mismatch. Expected {self.d_feat_stock_specific}, got {x_stock_specific.shape[2]}")


        # ---------------------------------------------------------------
        # Gating
        # ---------------------------------------------------------------
        # Apply gate using market info from the last timestep (most recent)
        gate_vals = self.feature_gate(x_market_info[:, -1, :])  # (N, d_feat_stock_specific)
        
        # Apply gating to stock-specific features across all timesteps
        # gate_vals: (N, d_feat_stock_specific), x_stock_specific: (N, T, d_feat_stock_specific)
        gated_stock_features = x_stock_specific * gate_vals.unsqueeze(1)  # broadcast to all T

        # ---------------------------------------------------------------
        # Main MASTER layers - OPTIMIZED: Direct module calls instead of Sequential
        # ---------------------------------------------------------------
        # Pass the gated stock-specific features through the model layers
        x = self.input_linear(gated_stock_features)  # (N, T, d_model)
        x = self.pos_encoding(x)  # (N, T, d_model)
        x = self.temporal_attention(x)  # (N, T, d_model)
        x = self.spatial_attention(x)  # (N, T, d_model)
        x = self.final_temporal_attention(x)  # (N, d_model)
        final_scores = self.output_linear(x)  # (N, 1)
        
        return final_scores

class MASTERModel():  # Remove SequenceModel inheritance
    def __init__(self, d_feat, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5, 
                 beta=5.0, gate_input_start_index=0, gate_input_end_index=1, n_epochs=100, lr=0.001, 
                 GPU=None, seed=None, save_path=None, save_prefix="master_model", 
                 loss_type="regrank", listfold_transformation="exponential"):
        """
        MASTER Model for Stock Ranking
        
        Args:
            loss_type: Type of loss function to use. Options:
                - "regrank": Original RegRankLoss (default)
                - "listfold": ListFold loss for long-short strategies  
                - "listfold_opt": Optimized ListFold loss
            listfold_transformation: Transformation for ListFold loss ("exponential" or "sigmoid")
        """

        # OPTIMIZED: Minimal initialization to avoid expensive base class setup
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if GPU is not None and torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.save_path = Path(save_path) if save_path else Path("model_output")
        self.save_prefix = save_prefix
        self.loss_type = loss_type
        self.listfold_transformation = listfold_transformation
        
        # FAST setup - only essential attributes without heavy operations
        self.early_stop = 20
        self.patience = 10
        self.decay_rate = 0.9
        self.min_lr = 1e-05
        self.metric = "loss"
        
        # Only set seed if needed - skip complex setup
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.seed)
        
        # Create save directory efficiently
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.d_feat_input_total = d_feat 
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.beta = beta 
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        # FAST model creation - no heavy .to() operations until needed
        self.model = PaperMASTERArchitecture(
            d_feat_input_total=self.d_feat_input_total,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta 
        )
        
        # Lazy initialization - only move to device when needed for first training
        self._device_moved = False
        
        # FAST optimizer setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # ENHANCED: Loss function setup with ListFold support
        if self.loss_type == "regrank":
            self.loss_fn = RegRankLoss(beta=self.beta)
            logger.info(f"Using RegRankLoss with beta={self.beta}")
        elif self.loss_type == "listfold":
            self.loss_fn = ListFoldLoss(transformation=self.listfold_transformation, beta=self.beta)
            logger.info(f"Using ListFoldLoss with transformation={self.listfold_transformation}, beta={self.beta}")
        elif self.loss_type == "listfold_opt":
            self.loss_fn = ListFoldLossOptimized(transformation=self.listfold_transformation, beta=self.beta)
            logger.info(f"Using ListFoldLossOptimized with transformation={self.listfold_transformation}, beta={self.beta}")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Choose from 'regrank', 'listfold', 'listfold_opt'")
        
        logger.info("MASTERModel (Fast Init) initialized.")
    
    def _ensure_device(self):
        """Lazy device transfer - only when actually needed"""
        if not self._device_moved:
            self.model = self.model.to(self.device)
            if self.device.type == 'cuda':
                self.loss_fn = self.loss_fn.to(self.device)
            self._device_moved = True


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
                 gate_input_start_index, gate_input_end_index, beta=5.0):
        super(MASTERNet, self).__init__()
        self.d_feat = d_feat
        self.d_model = d_model
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        # Calculate dimensions
        market_dim = gate_input_end_index - gate_input_start_index
        factor_dim = gate_input_start_index  # Only pre-market factors as per paper
        
        # Initialize components with corrected Gate
        self.gate = Gate(market_dim=market_dim, stock_feature_dim=factor_dim, beta=beta)
        self.linear_in = nn.Linear(factor_dim + market_dim, d_model)
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
            gate_weights_t = self.gate(market_features[:, t, :])  # (N, factor_dim)
            if factor_features.shape[2] > 0:
                # Apply gate to factor features at timestep t
                gated_factors_t = factor_features[:, t, :] * gate_weights_t
                combined_t = torch.cat([gated_factors_t, market_features[:, t, :]], dim=1)
            else:
                combined_t = market_features[:, t, :] * gate_weights_t[:, :market_features.shape[2]]
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
