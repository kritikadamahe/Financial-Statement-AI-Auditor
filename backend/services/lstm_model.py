import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # The hidden state from the last timestep is the context vector
        # We need to repeat it seq_len times to feed to the decoder
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We take the top layer hidden state
        context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        out, _ = self.decoder(context)
        
        return out

def compute_temporal_features(metrics_by_year: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Extracts raw sequences for the LSTM Autoencoder.
    Returns an array of shape (num_years, num_features).
    """
    years = sorted(metrics_by_year.keys())
    
    key_metrics = [
        "Revenue", "Cost of Goods Sold", "Net Income",
        "Current Assets", "Current Liabilities", "Total Assets",
        "Total Debt", "Total Equity", "Operating Cash Flow",
        "Gross Profit", "Operating Expenses"
    ]
    
    seq_data = []
    for year in years:
        m = metrics_by_year[year]
        year_features = []
        for metric in key_metrics:
            val = m.get(metric, 0.0)
            if np.isnan(val):
                val = 0.0
            year_features.append(float(val))
            
        # Add basic ratios per year to help the LSTM
        rev = year_features[0]
        cogs = year_features[1]
        ni = year_features[2]
        ca = year_features[3]
        cl = year_features[4]
        td = year_features[6]
        te = year_features[7]
        
        current_ratio = ca / cl if cl != 0 else 0.0
        debt_equity = td / te if te != 0 else 0.0
        net_margin = ni / rev if rev != 0 else 0.0
        gross_margin = (rev - cogs) / rev if rev != 0 else 0.0
        
        year_features.extend([current_ratio, debt_equity, net_margin, gross_margin])
        seq_data.append(year_features)
        
    return np.array(seq_data)
