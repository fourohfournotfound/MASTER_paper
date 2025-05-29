import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import pandas as pd
import numpy as np
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

# Setup logger for this module
logger = logging.getLogger(__name__)


# ============================================================================
# PAPER-ALIGNED MASTER IMPLEMENTATION - EXACT PAPER SPECIFICATIONS
# ============================================================================

class MarketStatusRepresentation:
    """
    EXACT MASTER Paper Implementation: Market Status Representation
    
    From paper: "We combine information from two aspects into a vector m_τ:
    (1) Market index price: current + historical average and std dev
    (2) Market index trading volume: average and std dev in past d' days"
    """
    
    def __init__(self, d_prime: int = 5):
        """
        Args:
            d_prime: Referable interval length for historical market information (paper uses 5,10,20,30,60)
        """
        self.d_prime = d_prime
    
    def construct_market_status(self, 
                              market_prices: pd.Series, 
                              market_volumes: pd.Series, 
                              current_date: pd.Timestamp) -> np.ndarray:
        """
        Construct market status vector m_τ as specified in the paper.
        
        Args:
            market_prices: Series of market index prices indexed by date
            market_volumes: Series of market index volumes indexed by date  
            current_date: Current timestamp τ
            
        Returns:
            Market status vector m_τ with shape [6]:
            [current_price, price_mean, price_std, volume_mean, volume_std, current_volume]
        """
        # Get historical window
        end_date = current_date
        start_date = current_date - pd.Timedelta(days=self.d_prime)
        
        # Filter to historical window
        price_window = market_prices[(market_prices.index >= start_date) & 
                                   (market_prices.index <= end_date)]
        volume_window = market_volumes[(market_volumes.index >= start_date) & 
                                     (market_volumes.index <= end_date)]
        
        # Current values
        current_price = market_prices.get(current_date, 0.0)
        current_volume = market_volumes.get(current_date, 0.0)
        
        # Historical statistics
        price_mean = price_window.mean() if len(price_window) > 0 else 0.0
        price_std = price_window.std() if len(price_window) > 1 else 0.0
        volume_mean = volume_window.mean() if len(volume_window) > 0 else 0.0
        volume_std = volume_window.std() if len(volume_window) > 1 else 0.0
        
        # Construct market status vector as per paper
        m_tau = np.array([
            current_price,
            price_mean, 
            price_std,
            volume_mean,
            volume_std,
            current_volume
        ], dtype=np.float32)
        
        return m_tau


class PaperGate(nn.Module):
    """
    EXACT Gate implementation from MASTER paper Equation 1:
    α(m_τ) = F · softmax_β(W_α m_τ + b_α)
    
    This replaces the previous Gate implementation with the exact paper specification.
    """
    
    def __init__(self, market_dim: int, stock_feature_dim: int, beta: float = 5.0):
        super().__init__()
        self.market_dim = market_dim
        self.stock_feature_dim = stock_feature_dim
        self.beta = beta
        
        # Linear transformation W_α and bias b_α (exact paper implementation)
        self.W_alpha = nn.Linear(market_dim, stock_feature_dim)
        
    def forward(self, market_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_info: Market status m_τ with shape (batch_size, market_dim)
        
        Returns:
            Gating coefficients α(m_τ) with shape (batch_size, stock_feature_dim)
        """
        # Linear transformation: W_α m_τ + b_α
        logits = self.W_alpha(market_info)  # (batch_size, stock_feature_dim)
        
        # Temperature-scaled softmax (exact paper equation)
        probabilities = F.softmax(logits / self.beta, dim=-1)
        
        # Scale by F (number of features) as per paper equation
        alpha_values = self.stock_feature_dim * probabilities
        
        return alpha_values


class PaperPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in MASTER paper"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class PaperIntraStockAggregation(nn.Module):
    """
    MASTER Paper: Intra-Stock Aggregation (Component 2)
    
    "We apply a bi-directional sequential encoder to obtain the local output 
    at each time step t. We instantiate the sequential encoder with a single-layer 
    transformer encoder."
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention for intra-stock aggregation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: Feature embeddings (N_stocks, T_lookback, d_model)
            
        Returns:
            H: Local embeddings (N_stocks, T_lookback, d_model)
        """
        # Self-attention within each stock's sequence
        Y_norm = self.norm1(Y)
        attn_output, _ = self.self_attention(Y_norm, Y_norm, Y_norm)
        Y = Y + attn_output  # Residual connection
        
        # Feed-forward network
        Y_norm2 = self.norm2(Y)
        ffn_output = self.ffn(Y_norm2)
        H = Y + ffn_output  # Residual connection
        
        return H


class PaperInterStockAggregation(nn.Module):
    """
    MASTER Paper: Inter-Stock Aggregation (Component 3)
    
    "At each time step, we gather the local embedding of all stocks and perform 
    multi-head attention mechanism to mine asymmetric and dynamic inter-stock correlation."
    
    KEY: This processes each timestep separately for "momentary correlation"
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention for inter-stock aggregation
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Local embeddings (N_stocks, T_lookback, d_model)
            
        Returns:
            Z: Temporal embeddings (N_stocks, T_lookback, d_model)
        """
        N_stocks, T_lookback, d_model = H.shape
        Z_list = []
        
        # Process each time step separately for momentary correlation (paper's key innovation)
        for t in range(T_lookback):
            # Get embeddings at time step t for all stocks
            H_t = H[:, t, :].unsqueeze(1)  # (N_stocks, 1, d_model)
            
            # Normalize
            H_t_norm = self.norm1(H_t)
            
            # Cross-attention across stocks at time step t
            attn_output, _ = self.cross_attention(H_t_norm, H_t_norm, H_t_norm)
            H_t = H_t + attn_output  # Residual connection
            
            # Feed-forward network
            H_t_norm2 = self.norm2(H_t)
            ffn_output = self.ffn(H_t_norm2)
            Z_t = H_t + ffn_output  # Residual connection
            
            Z_list.append(Z_t.squeeze(1))  # (N_stocks, d_model)
        
        # Stack to get (N_stocks, T_lookback, d_model)
        Z = torch.stack(Z_list, dim=1)
        
        return Z


class PaperTemporalAggregation(nn.Module):
    """
    MASTER Paper: Temporal Aggregation (Component 4)
    
    "We use the latest temporal embedding z_{u,τ} as the query vector, 
    and compute the attention score λ_{u,t}"
    
    Implements exact paper equation for temporal attention.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        # Transformation matrix W_λ (exact paper implementation)
        self.W_lambda = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: Temporal embeddings (N_stocks, T_lookback, d_model)
            
        Returns:
            e: Comprehensive stock embeddings (N_stocks, d_model)
        """
        # Use latest temporal embedding as query (paper specification)
        query = Z[:, -1, :].unsqueeze(-1)  # (N_stocks, d_model, 1)
        
        # Transform temporal embeddings
        H = self.W_lambda(Z)  # (N_stocks, T_lookback, d_model)
        
        # Compute attention scores (exact paper equation)
        attention_scores = torch.bmm(H, query).squeeze(-1)  # (N_stocks, T_lookback)
        attention_weights = F.softmax(attention_scores, dim=1)  # (N_stocks, T_lookback)
        
        # Weighted aggregation
        attention_weights = attention_weights.unsqueeze(1)  # (N_stocks, 1, T_lookback)
        e = torch.bmm(attention_weights, Z).squeeze(1)  # (N_stocks, d_model)
        
        return e


class PaperMASTERArchitecture(nn.Module):
    """
    EXACT MASTER Paper Architecture Implementation
    
    The 5-component architecture as described in the paper:
    1. Market-Guided Gating
    2. Intra-Stock Aggregation
    3. Inter-Stock Aggregation 
    4. Temporal Aggregation
    5. Prediction
    
    This replaces PaperMASTERArchitecture to be the main implementation.
    """
    
    def __init__(self,
                 d_feat_stock: int,
                 market_status_dim: int = 6,  # As per paper specification
                 d_model: int = 256,
                 t_nhead: int = 4,
                 s_nhead: int = 2,
                 dropout: float = 0.1,
                 beta: float = 5.0):
        super().__init__()
        
        self.d_feat_stock = d_feat_stock
        self.market_status_dim = market_status_dim
        self.d_model = d_model
        
        # Component 1: Market-Guided Gating (exact paper implementation)
        self.gate = PaperGate(
            market_dim=market_status_dim,
            stock_feature_dim=d_feat_stock,
            beta=beta
        )
        
        # Feature encoder: transform to embedding space
        self.feature_encoder = nn.Linear(d_feat_stock, d_model)
        
        # Positional encoding
        self.pos_encoding = PaperPositionalEncoding(d_model)
        
        # Component 2: Intra-Stock Aggregation
        self.intra_stock_aggregation = PaperIntraStockAggregation(
            d_model=d_model,
            nhead=t_nhead,
            dropout=dropout
        )
        
        # Component 3: Inter-Stock Aggregation
        self.inter_stock_aggregation = PaperInterStockAggregation(
            d_model=d_model,
            nhead=s_nhead,
            dropout=dropout
        )
        
        # Component 4: Temporal Aggregation
        self.temporal_aggregation = PaperTemporalAggregation(d_model)
        
        # Component 5: Prediction
        self.predictor = nn.Linear(d_model, 1)
        
    def forward(self, 
                stock_features: torch.Tensor,
                market_status: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stock_features: Stock-specific features (N_stocks, T_lookback, d_feat_stock)
            market_status: Market status vector (N_stocks, market_status_dim)
            
        Returns:
            predictions: Stock predictions (N_stocks, 1)
        """
        # Component 1: Market-Guided Gating
        gate_weights = self.gate(market_status)  # (N_stocks, d_feat_stock)
        
        # Apply gating to stock features across all time steps
        gated_features = stock_features * gate_weights.unsqueeze(1)
        
        # Transform to embedding space
        Y = self.feature_encoder(gated_features)  # (N_stocks, T_lookback, d_model)
        
        # Add positional encoding
        Y = self.pos_encoding(Y)
        
        # Component 2: Intra-Stock Aggregation
        H = self.intra_stock_aggregation(Y)  # (N_stocks, T_lookback, d_model)
        
        # Component 3: Inter-Stock Aggregation
        Z = self.inter_stock_aggregation(H)  # (N_stocks, T_lookback, d_model)
        
        # Component 4: Temporal Aggregation
        e = self.temporal_aggregation(Z)  # (N_stocks, d_model)
        
        # Component 5: Prediction
        predictions = self.predictor(e)  # (N_stocks, 1)
        
        return predictions


# ============================================================================
# LEGACY COMPONENTS (PRESERVED FOR COMPATIBILITY)
# ============================================================================

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


class MASTERModel():  # Remove SequenceModel inheritance
    def __init__(self, d_feat, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5, 
                 beta=5.0, gate_input_start_index=0, gate_input_end_index=1, n_epochs=100, lr=0.001, 
                 GPU=None, seed=None, save_path=None, save_prefix="master_model", 
                 loss_type="mse", listfold_transformation="exponential", use_paper_architecture=True):
        """
        MASTER Model for Stock Ranking - NOW PAPER-ALIGNED BY DEFAULT
        
        Args:
            loss_type: Type of loss function to use. Options:
                - "mse": MSE loss (MASTER paper default)
                - "regrank": Original RegRankLoss  
                - "listfold": ListFold loss for long-short strategies  
                - "listfold_opt": Optimized ListFold loss
            listfold_transformation: Transformation for ListFold loss ("exponential" or "sigmoid")
            use_paper_architecture: Whether to use paper-aligned architecture (default: True)
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
        self.use_paper_architecture = use_paper_architecture
        
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
        
        # FAST model creation - use paper-aligned architecture by default
        if use_paper_architecture:
            # Calculate market status dimension (6-dimensional as per paper)
            market_status_dim = 6  # Fixed: current_price, price_mean, price_std, volume_mean, volume_std, current_volume
            
            # Calculate actual stock feature dimension (total features minus market features)
            market_features_count = gate_input_end_index - gate_input_start_index
            stock_feature_dim = self.d_feat_input_total - market_features_count
            
            self.model = PaperMASTERArchitecture(
                d_feat_stock=stock_feature_dim,  # FIXED: Use actual stock features count
                market_status_dim=market_status_dim,
                d_model=self.d_model,
                t_nhead=self.t_nhead,
                s_nhead=self.s_nhead,
                dropout=self.T_dropout_rate,
                beta=self.beta
            )
            
            # Store gate indices on the model for easy access during batch processing
            self.model.gate_input_start_index = gate_input_start_index
            self.model.gate_input_end_index = gate_input_end_index
            
            logger.info("Using Paper-Aligned MASTER Architecture (exact paper implementation)")
        else:
            # Legacy architecture (preserved for compatibility)
            # Calculate actual stock feature dimension (total features minus market features)
            market_features_count = gate_input_end_index - gate_input_start_index
            stock_feature_dim = self.d_feat_input_total - market_features_count
            
            self.model = PaperMASTERArchitecture(
                d_feat_stock=stock_feature_dim,  # FIXED: Use actual stock features count
                market_status_dim=self.gate_input_end_index - self.gate_input_start_index,
                d_model=self.d_model,
                t_nhead=self.t_nhead,
                s_nhead=self.s_nhead,
                dropout=self.T_dropout_rate,
                beta=self.beta
            )
            
            # Store gate indices on the model for easy access during batch processing
            self.model.gate_input_start_index = gate_input_start_index
            self.model.gate_input_end_index = gate_input_end_index
        
        # Lazy initialization - only move to device when needed for first training
        self._device_moved = False
        
        # FAST optimizer setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # ENHANCED: Loss function setup with paper-aligned MSE as default
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
            logger.info(f"Using MSE Loss (MASTER paper default)")
        elif self.loss_type == "regrank":
            self.loss_fn = RegRankLoss(beta=self.beta)
            logger.info(f"Using RegRankLoss with beta={self.beta}")
        elif self.loss_type == "listfold":
            self.loss_fn = ListFoldLoss(transformation=self.listfold_transformation, beta=self.beta)
            logger.info(f"Using ListFoldLoss with transformation={self.listfold_transformation}, beta={self.beta}")
        elif self.loss_type == "listfold_opt":
            self.loss_fn = ListFoldLossOptimized(transformation=self.listfold_transformation, beta=self.beta)
            logger.info(f"Using ListFoldLossOptimized with transformation={self.listfold_transformation}, beta={self.beta}")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Choose from 'mse', 'regrank', 'listfold', 'listfold_opt'")
        
        logger.info("MASTERModel (Paper-Aligned) initialized.")
    
    def _ensure_device(self):
        """Lazy device transfer - only when actually needed"""
        if not self._device_moved:
            self.model = self.model.to(self.device)
            if self.device.type == 'cuda':
                self.loss_fn = self.loss_fn.to(self.device)
            self._device_moved = True

    def forward_paper_aligned(self, stock_features: torch.Tensor, market_status: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using paper-aligned architecture with separated inputs
        
        Args:
            stock_features: Stock-specific features (N_stocks, T_lookback, d_feat_stock)
            market_status: Market status vector (N_stocks, market_status_dim)
            
        Returns:
            predictions: Stock predictions (N_stocks,) or (N_stocks, 1)
        """
        self._ensure_device()
        return self.model(stock_features, market_status)

    def predict_paper_aligned(self, stock_features: torch.Tensor, market_status: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using paper-aligned architecture
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.forward_paper_aligned(stock_features, market_status)
        return predictions.squeeze() if predictions.dim() > 1 else predictions


# ============================================================================
# PAPER-ALIGNED MARKET STATUS UTILITIES
# ============================================================================

def prepare_market_index_data(df: pd.DataFrame) -> tuple:
    """
    Prepare market index price and volume series from the dataset as per MASTER paper
    
    Args:
        df: Multi-index DataFrame with (ticker, date) index
        
    Returns:
        market_prices: Series of daily market index prices (cross-sectional mean)
        market_volumes: Series of daily market index volumes (cross-sectional mean)
    """
    # Calculate daily market index as cross-sectional means (paper method)
    daily_market_data = df.groupby(level='date').agg({
        'closeadj': 'mean',  # Market index price
        'volume': 'mean'     # Market index volume
    })
    
    market_prices = daily_market_data['closeadj']
    market_volumes = daily_market_data['volume']
    
    return market_prices, market_volumes


def normalize_returns_per_day(targets: np.ndarray, dates: list) -> np.ndarray:
    """
    Normalize returns using daily z-score as specified in the MASTER paper:
    r_u = Norm_S(ř_u) where ř_u is the raw return ratio
    
    This replaces global normalization with paper-specified daily z-score normalization.
    """
    df_temp = pd.DataFrame({
        'date': dates,
        'target': targets
    })
    
    # Group by date and normalize (daily z-score as per paper)
    def zscore_normalize(group):
        return (group - group.mean()) / (group.std() + 1e-8)
    
    normalized = df_temp.groupby('date')['target'].transform(zscore_normalize)
    return normalized.values
