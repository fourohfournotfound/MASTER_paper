import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

from base_model import SequenceModel


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
                 save_path='model/', save_prefix='master_model'):
        
        # Initialize the parent SequenceModel class
        super().__init__(n_epochs=n_epochs, lr=lr, GPU=GPU, seed=seed,
                         train_stop_loss_thred=train_stop_loss_thred,
                         save_path=save_path, save_prefix=save_prefix)

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

    # The fit and predict methods are inherited from SequenceModel.
    # If MASTER needs specific logic for training or prediction beyond what SequenceModel provides,
    # you can override those methods here. For now, we assume SequenceModel's methods are sufficient.

    def train_predict(self, train_data, test_data=None):
        """
        A convenience method to encapsulate training and prediction.
        train_data: DataLoader or Dataset for training.
        test_data: DataLoader or Dataset for testing.
        """
        print(f"[MASTER] Starting training for {self.n_epochs} epochs.")
        self.fit(dl_train=train_data, dl_valid=test_data) # dl_valid can be test_data for simplicity here
        
        predictions = None
        metrics = None
        if test_data:
            print("[MASTER] Starting prediction on test data.")
            predictions, metrics = self.predict(dl_test=test_data)
            print(f"[MASTER] Test Metrics: {metrics}")
        
        print("[MASTER] train_predict finished.")
        return predictions, metrics


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
