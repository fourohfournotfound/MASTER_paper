import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiIndexDataset(Dataset):
    """Dataset for multi-index time series with ticker and date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by ['ticker', 'date'] (order insensitive).
    lookback : int, optional
        Number of past timesteps used as features. Default 8.
    feature_cols : list[str], optional
        Columns used as features. If None, all columns except `return_col` are used.
    return_col : str, optional
        Column used for return calculation. Default 'closeadj'.
    """
    is_multiindex_dataset = True

    def __init__(self, df, lookback=8, feature_cols=None, return_col="closeadj"):
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("df must have a MultiIndex with ['ticker','date']")
        self.lookback = lookback
        # ensure index order ticker/date
        if list(df.index.names) != ['ticker', 'date']:
            df = df.reorder_levels(['ticker', 'date']).sort_index()
        else:
            df = df.sort_index()
        self.df = df
        self.return_col = return_col
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != return_col]
        self.feature_cols = feature_cols
        self.dates = sorted(df.index.get_level_values('date').unique())
        self.tickers = df.index.get_level_values('ticker').unique()
        self.samples = []
        self.index_tuples = []
        for i in range(self.lookback, len(self.dates)-1):
            cur_date = self.dates[i]
            prev_dates = self.dates[i-self.lookback:i]
            next_date = self.dates[i+1]
            day_arrays = []
            day_index = []
            for t in self.tickers:
                tdf = df.xs(t, level='ticker')
                if set(prev_dates+[cur_date,next_date]).issubset(tdf.index):
                    feats = tdf.loc[prev_dates, self.feature_cols].values
                    ret = tdf.loc[next_date, self.return_col] / tdf.loc[cur_date, self.return_col] - 1
                    arr = np.zeros((self.lookback, len(self.feature_cols)+1), dtype=np.float32)
                    arr[:, :-1] = feats
                    arr[-1, -1] = ret
                    day_arrays.append(arr)
                    day_index.append((next_date, t))
            if day_arrays:
                self.samples.append(np.stack(day_arrays))
                self.index_tuples.extend(day_index)
        if self.index_tuples:
            self.index = pd.MultiIndex.from_tuples(self.index_tuples, names=['date', 'ticker'])
        else:
            self.index = pd.MultiIndex.from_tuples([], names=['date', 'ticker'])
        self.n_features = len(self.feature_cols)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx])

    def get_index(self):
        return self.index
