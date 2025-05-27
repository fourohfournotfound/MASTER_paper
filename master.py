import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import torch.nn.functional as F

# Assuming base_model.py is in the same directory or accessible in PYTHONPATH
from base_model import SequenceModel 
# If running main_multi_index.py from MASTER_paper folder, then:
# from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000): # Increased max_len for typical sequence lengths
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) # Original did not have this, but common to make it batch-first ready if needed
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (max_len, d_model)
        # We need to add pe[:x.shape[1], :] to x
        return x + self.pe[:x.size(1), :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        # In SAttention, input x is (batch_size * num_stocks_in_batch, 1, d_model) 
        # but effectively it's processing across stocks for a given time step.
        # The original MASTER paper's SAttention processes (N, S, D) where N is batch(days), S is stocks, D is d_model.
        # Here, input after TAttention and TemporalAttention is (batch_size, d_model).
        # The paper implies SAttention operates on a (Batch_of_Days, Num_Stocks, Features) tensor.
        # This implementation seems more like a standard self-attention block.
        # For the current pipeline with (Batch_of_Sequences, Seq_Len, Features), 
        # if SAttention is meant for inter-stock, the input needs reshaping or context.
        # Given the provided structure, it acts like another transformer encoder layer.
        # Let's assume it processes the output of TAttention which is (Batch, SeqLen, d_model).

        self.temperature = math.sqrt(self.d_model/nhead) # Should be sqrt(dim_k_per_head)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.out_trans = nn.Linear(d_model, d_model) # Added output transformation

        self.attn_dropout_rate = dropout
        # self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)]) # More typical

        # LayerNorms
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        
        self.ffn = nn.Sequential(
            Linear(d_model, d_model * 4), # Expanded FFN
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model * 4, d_model), # Expanded FFN
            Dropout(p=dropout)
        )

    def forward(self, x_input): # x shape: (Batch, NumStocks_or_SeqLen, Dim)
        x = self.norm1(x_input)
        
        # Reshape for multi-head attention: (Batch, NumHeads, NumStocks_or_SeqLen, DimPerHead)
        B, S, D = x.shape
        D_per_head = D // self.nhead

        q = self.qtrans(x).reshape(B, S, self.nhead, D_per_head).transpose(1, 2) # (B, nH, S, DpH)
        k = self.ktrans(x).reshape(B, S, self.nhead, D_per_head).transpose(1, 2) # (B, nH, S, DpH)
        v = self.vtrans(x).reshape(B, S, self.nhead, D_per_head).transpose(1, 2) # (B, nH, S, DpH)

        # Scaled Dot-Product Attention
        # (B, nH, S, DpH) x (B, nH, DpH, S) -> (B, nH, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_per_head)
        attn_weights = torch.softmax(scores, dim=-1)
        
        if self.attn_dropout_rate > 0:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout_rate, training=self.training)

        # (B, nH, S, S) x (B, nH, S, DpH) -> (B, nH, S, DpH)
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).reshape(B, S, D) # (B, S, D)
        att_output = self.out_trans(context)

        # Add & Norm (Residual connection)
        xt = x_input + att_output # x_input is before norm1
        # FFN
        att_output_ffn = self.ffn(self.norm2(xt))
        final_output = xt + att_output_ffn

        return final_output


class TAttention(nn.Module): # Temporal Self-Attention
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Using nn.MultiheadAttention for simplicity and correctness
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # LayerNorms
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model * 4), # Common practice to expand FFN
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model * 4, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x): # x shape: (Batch, SeqLen, Dim)
        # Pre-LayerNorm (common variant)
        x_norm = self.norm1(x)
        
        # Self-attention
        # attn_output, _ = self.multihead_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        # For PyTorch <1.9, need_weights defaults to True if not given. For >=1.9, it's False.
        # For safety, explicitly state if you don't need weights.
        attn_output, _ = self.multihead_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)


        # Add & Norm (Residual connection)
        x = x + attn_output # Residual connection from original x
        
        # FFN part
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output # Residual connection

        return x


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output # This seems to be used for scaling output, unusual.
                                # Typically gate output is directly used as weights (0-1 range).
        self.t = beta # Temperature for softmax

    def forward(self, gate_input): # gate_input shape: (Batch, d_input)
        output = self.trans(gate_input) # (Batch, d_output)
        # Softmax over the feature dimension of the gate's output
        output_probs = torch.softmax(output / self.t, dim=-1) # (Batch, d_output)
        
        # The original multiplies by self.d_output. If d_output is the number of primary features,
        # this scaling makes each element of output_probs effectively contribute more if d_output is large.
        # A standard feature gate would output values in [0,1] or similar to directly multiply features.
        # Keeping original behavior:
        return self.d_output * output_probs


class TemporalAttentionPool(nn.Module): # Renamed from original TemporalAttention to avoid confusion
    def __init__(self, d_model):
        super().__init__()
        # This layer computes attention weights over the sequence to get a weighted sum.
        self.trans = nn.Linear(d_model, d_model, bias=False) # Computes "keys" or "values" for attention
        self.query_vec = nn.Parameter(torch.empty(d_model).uniform_(-0.1, 0.1)) # Learnable query vector

    def forward(self, z): # z shape: (Batch, SeqLen, Dim)
        # Project z to get "keys" for attention mechanism
        h = torch.tanh(self.trans(z)) # (Batch, SeqLen, Dim), adding non-linearity
        
        # Compute attention scores: (Batch, SeqLen, Dim) * (Dim, 1) -> (Batch, SeqLen, 1)
        # query_vec is (Dim), unsqueeze to (Dim, 1) for matmul, or (1, 1, Dim) for bmm with h
        # Let's use a dot product similarity with the learnable query
        # query expanded: (1, 1, Dim)
        scores = torch.matmul(h, self.query_vec.unsqueeze(-1)).squeeze(-1) # (Batch, SeqLen)
        
        lam = torch.softmax(scores, dim=1) # (Batch, SeqLen), attention weights for each time step
        
        # Weighted sum: lam.unsqueeze(1) is (Batch, 1, SeqLen)
        # (Batch, 1, SeqLen) x (Batch, SeqLen, Dim) -> (Batch, 1, Dim)
        output = torch.bmm(lam.unsqueeze(1), z).squeeze(1) # (Batch, Dim)
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, 
                 gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        d_gate_input_features = (gate_input_end_index - gate_input_start_index)
        
        self.feature_gate = Gate(d_gate_input_features, d_feat, beta=beta)

        self.input_projection = nn.Linear(d_feat, d_model) # Project primary features to d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Intra-stock temporal aggregation (Transformer Encoder Layer style)
        self.temporal_attention_layer = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        
        # Inter-stock aggregation (if data is shaped N, S, D) or another self-attention layer
        # Given current data flow (N, T, D), SAttention will act as another temporal self-attention layer
        # or a self-attention over features if applied differently.
        # The paper's diagram suggests SAttention operates *after* some form of per-stock aggregation.
        # If input to SAttention is (Batch_Size, Num_Stocks_or_Items, d_model)
        self.spatial_attention_layer = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        
        # Final temporal pooling over the sequence
        self.temporal_pooling_layer = TemporalAttentionPool(d_model=d_model)
        
        self.decoder = nn.Linear(d_model, 1) # Predict a single value (e.g., future return)

    def forward(self, x): # x shape: (Batch, SeqLen, TotalFeatures)
        # TotalFeatures = d_feat_primary + d_gate_features
        
        # Primary features for the sequence
        primary_features_seq = x[:, :, :self.gate_input_start_index] # (Batch, SeqLen, d_feat_primary)
        
        # Gate input features: from the *last time step* of the designated gate feature slice
        gate_input_slice = x[:, -1, self.gate_input_start_index:self.gate_input_end_index] # (Batch, d_gate_features)
        
        # Get feature weights from the gate
        feature_weights = self.feature_gate(gate_input_slice) # (Batch, d_feat_primary)
        
        # Apply gate: unsqueeze feature_weights to (Batch, 1, d_feat_primary) for broadcasting
        gated_features = primary_features_seq * feature_weights.unsqueeze(1) # Element-wise multiplication
        
        # Process gated features
        projected_features = self.input_projection(gated_features) # (Batch, SeqLen, d_model)
        pos_encoded_features = self.pos_encoder(projected_features)
        
        temporal_out = self.temporal_attention_layer(pos_encoded_features) # (Batch, SeqLen, d_model)
        
        # The original MASTER paper applies SAttention across stocks.
        # If `temporal_out` is (Batch, SeqLen, d_model), and SAttention expects (Batch, NumItems, d_model)
        # where NumItems could be SeqLen here, it acts as another self-attention over the sequence.
        # Or, if data was (Batch_of_Days, NumStocks, SeqLen, Feats) and aggregated to (Batch_of_Days, NumStocks, d_model)
        # then SAttention would work across stocks.
        # With current (Batch_of_sequences, SeqLen, d_model), SAttention acts on SeqLen.
        spatial_out = self.spatial_attention_layer(temporal_out) # (Batch, SeqLen, d_model)
        
        pooled_out = self.temporal_pooling_layer(spatial_out) # (Batch, d_model)
        
        output = self.decoder(pooled_out) # (Batch, 1)
        return output.squeeze(-1) # (Batch)


class MASTERModel(SequenceModel):
    def __init__(
            self, 
            d_feat, # Number of primary input features (e.g., 6 from CSV, or 158 in original paper)
            d_model, 
            t_nhead, 
            s_nhead, 
            gate_input_start_index, # Start index for gate features in the combined input tensor X
            gate_input_end_index,   # End index for gate features
            T_dropout_rate, 
            S_dropout_rate, 
            beta, 
            **kwargs, # Catches n_epochs, lr, GPU, seed, etc. for SequenceModel
    ):
        super(MASTERModel, self).__init__(**kwargs) # Pass kwargs to SequenceModel
        
        self.d_feat = d_feat # Number of primary features (output dim of Gate)
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.beta = beta

        self.init_model() # Initialize the actual PyTorch model

    def init_model(self):
        # This method is called by SequenceModel's __init__ if not overridden,
        # but we define it here to construct the MASTER module.
        self.model = MASTER(
            d_feat=self.d_feat, 
            d_model=self.d_model, 
            t_nhead=self.t_nhead, 
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate, 
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index, 
            beta=self.beta
        )
        # Call super().init_model() AFTER self.model is defined,
        # so it can set up optimizer and move to device.
        super(MASTERModel, self).init_model()
