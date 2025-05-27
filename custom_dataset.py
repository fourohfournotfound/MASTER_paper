import pandas as pd
import numpy as np
from io import StringIO

class MultiIndexTimeSeriesDataset:
    def __init__(self, csv_path_or_buffer, lookback_window):
        self.lookback_window = lookback_window
        
        # Load data
        if isinstance(csv_path_or_buffer, str):
            self.df = pd.read_csv(csv_path_or_buffer)
        else: # Assuming it's a buffer like StringIO
            self.df = pd.read_csv(csv_path_or_buffer)

        # Date Parsing
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Feature Engineering - lastupdated
        try:
            self.df['lastupdated_numeric'] = pd.to_datetime(self.df['lastupdated']).astype(np.int64) // 10**9 # Seconds since epoch
            print("Successfully converted 'lastupdated' to numeric (seconds since epoch).")
        except Exception as e:
            print(f"Warning: Could not convert 'lastupdated' to datetime: {e}. Keeping it as is or dropping if it causes issues.")
            # Decide whether to keep self.df['lastupdated'] or drop it. For now, let's try keeping original if conversion fails
            # and ensure it's not selected as a numeric feature if it's not numeric.
            # If 'lastupdated_numeric' was not created, it won't be in numeric features.
            # If original 'lastupdated' is non-numeric, it will be excluded by select_dtypes(include=np.number) later or cause error
            if 'lastupdated_numeric' not in self.df.columns:
                 # If you want to ensure 'lastupdated' is dropped if not convertible:
                 # self.df = self.df.drop(columns=['lastupdated'], errors='ignore')
                 # print("Dropped 'lastupdated' column as it could not be converted to numeric.")
                 pass


        # Set Index
        self.df = self.df.set_index(['ticker', 'date']).sort_index()

        # Feature Selection
        # All columns except original 'ticker' and 'date' (now index).
        # 'closeadj' is a feature. 'lastupdated_numeric' (if created) is a feature.
        # Original 'lastupdated' might be non-numeric, handle that.
        
        potential_feature_columns = [col for col in self.df.columns if col not in ['ticker', 'date']]
        
        # Ensure only numeric features are selected if any non-numeric ones remain (e.g. original lastupdated)
        self.feature_names = self.df[potential_feature_columns].select_dtypes(include=np.number).columns.tolist()
        
        if 'lastupdated' in self.df.columns and 'lastupdated_numeric' not in self.feature_names and 'lastupdated' in self.feature_names:
            # If original 'lastupdated' is still a feature name and it's not numeric, remove it.
            if not pd.api.types.is_numeric_dtype(self.df['lastupdated']):
                print(f"Warning: Original 'lastupdated' column is non-numeric and was not converted. Removing from features.")
                self.feature_names.remove('lastupdated')
        
        self.d_feat = len(self.feature_names)
        print(f"Selected features ({self.d_feat}): {self.feature_names}")


        # Label Calculation
        self.df['label'] = self.df.groupby('ticker')['closeadj'].transform(
            lambda x: (x.shift(-1) - x) / x
        )

        # Data Segmentation
        self.features = []
        self.labels = []
        self.index_tuples = []

        grouped_by_ticker = self.df.groupby('ticker')
        for ticker_name, group_data in grouped_by_ticker:
            # Ensure group_data features are numeric and handle NaNs before windowing
            # Fill NaNs or drop rows. For simplicity, let's fill with 0 for features, but this might need better handling.
            # Labels with NaN (last day of sequence) are handled by dropping those sequences.
            
            numeric_features_df = group_data[self.feature_names].fillna(0) # Example: fillna with 0
            labels_series = group_data['label']

            for i in range(len(group_data) - self.lookback_window): # Ensure there's a label for the last day of window
                feature_sequence = numeric_features_df.iloc[i : i + self.lookback_window].values
                
                # Label corresponds to the last day of the sequence
                # The label for date D is calculated using closeadj of D and D+1
                # So, for a sequence ending at D_t, the label is for D_t (predicting D_t+1's return)
                label = labels_series.iloc[i + self.lookback_window -1] # Label for the last day of the window
                
                current_date = group_data.index[i + self.lookback_window -1][1] # date is the second part of multi-index

                if not np.isnan(label): # Only add if label is not NaN
                    self.features.append(feature_sequence)
                    self.labels.append(label)
                    self.index_tuples.append((current_date, ticker_name))
                # else:
                #     print(f"Skipping sequence ending on {current_date} for {ticker_name} due to NaN label (likely end of data).")


    def get_index(self):
        if not self.index_tuples:
            return pd.MultiIndex(levels=[[], []], codes=[[], []], names=['datetime', 'instrument'])
        
        multi_idx = pd.MultiIndex.from_tuples(self.index_tuples, names=['datetime', 'instrument'])
        
        # Sort the MultiIndex: datetime first, then instrument
        # To do this, convert to DataFrame, sort, then back to MultiIndex
        df_idx = multi_idx.to_frame(index=False)
        df_idx_sorted = df_idx.sort_values(by=['datetime', 'instrument'])
        
        return pd.MultiIndex.from_frame(df_idx_sorted)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

if __name__ == '__main__':
    # Dummy CSV data based on user issue description
    # Added more variety in dates and tickers
    # 'lastupdated' has mixed formats to test conversion
    csv_data_string = """ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated
T1,2023-01-01,10,11,9,10.5,1000,10.5,10.5,2023-01-01
T1,2023-01-02,10.5,11.5,10.2,11.0,1200,11.0,11.0,2023-01-02
T1,2023-01-03,11.0,11.2,10.8,11.1,800,11.1,11.1,03-01-2023
T1,2023-01-04,11.1,12.0,11.0,11.5,1500,11.5,11.5,2023-01-04
T1,2023-01-05,11.5,11.7,11.3,11.6,1100,11.6,11.6,2023-01-05
T1,2023-01-06,11.6,12.2,11.5,12.0,1300,12.0,12.0,2023-01-06
T1,2023-01-07,12.0,12.5,11.8,12.2,900,12.2,12.2,2023-01-07
T1,2023-01-08,12.2,12.8,12.1,12.5,1400,12.5,12.5,2023-01-08
T1,2023-01-09,12.5,13.0,12.4,12.8,1600,12.8,12.8,2023-01-09
T1,2023-01-10,12.8,13.5,12.7,13.0,1700,13.0,13.0,2023-01-10
T2,2023-01-01,100,101,99,100.5,2000,100.5,100.5,InvalidDate
T2,2023-01-02,100.5,102.5,100.2,101.0,2200,101.0,101.0,2023-01-02
T2,2023-01-03,101.0,101.5,100.8,101.2,1800,101.2,101.2,2023-01-03
T2,2023-01-04,101.2,103.0,101.0,102.0,2500,102.0,102.0,2023-01-04
T2,2023-01-05,102.0,102.5,101.5,102.3,2100,102.3,102.3,2023-01-05
T2,2023-01-06,102.3,103.5,102.1,103.0,2300,103.0,103.0,2023-01-06
T2,2023-01-07,103.0,104.0,102.8,103.5,1900,103.5,103.5,2023-01-07
T2,2023-01-08,103.5,105.0,103.2,104.0,2400,104.0,104.0,08/01/2023
T2,2023-01-09,104.0,106.0,103.8,105.0,2600,105.0,105.0,2023-01-09
T2,2023-01-10,105.0,107.0,104.5,106.0,2700,106.0,106.0,2023-01-10
"""
    csv_file_buffer = StringIO(csv_data_string)
    
    lookback = 8
    dataset = MultiIndexTimeSeriesDataset(csv_path_or_buffer=csv_file_buffer, lookback_window=lookback)

    print(f"\nNumber of features (d_feat): {dataset.d_feat}")
    print(f"Total number of sequences: {len(dataset)}")

    if len(dataset) > 0:
        print("\nOutput of get_index().head(10):")
        print(dataset.get_index().to_frame().head(10)) # .to_frame() for better printing of MultiIndex
        print("\nOutput of get_index().tail(10):")
        print(dataset.get_index().to_frame().tail(10))

        first_features, first_label = dataset[0]
        print(f"\nShape of the first feature sequence: {first_features.shape}")
        print(f"Label for the first sequence: {first_label}")

        last_features, last_label = dataset[len(dataset)-1]
        print(f"\nShape of the last feature sequence: {last_features.shape}")
        print(f"Label for the last sequence: {last_label}")
    else:
        print("\nNo sequences were generated. Check data and lookback window.")

import pandas as pd
import numpy as np
from io import StringIO

class MultiIndexTimeSeriesDataset:
    def __init__(self, csv_path_or_buffer, lookback_window):
        self.lookback_window = lookback_window
        
        # Load data
        if isinstance(csv_path_or_buffer, str):
            self.df = pd.read_csv(csv_path_or_buffer)
        else: # Assuming it's a buffer like StringIO
            self.df = pd.read_csv(csv_path_or_buffer)

        # Date Parsing
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Feature Engineering - lastupdated
        try:
            # Attempt to convert 'lastupdated' to datetime, coercing errors
            self.df['lastupdated_dt'] = pd.to_datetime(self.df['lastupdated'], errors='coerce')
            # Convert valid datetimes to numeric (seconds since epoch)
            self.df['lastupdated_numeric'] = self.df['lastupdated_dt'].astype(np.int64) // 10**9
            # For rows where conversion failed (NaT), 'lastupdated_numeric' will be NaT.astype(int64) which is a large negative number
            # We should handle these NaNs/NaTs, e.g., by filling with a specific value or using a mask
            if self.df['lastupdated_numeric'].isnull().any() or (self.df['lastupdated_dt'].isnull().sum() > 0) :
                 print(f"Warning: Some 'lastupdated' values could not be converted to datetime. These will be NaNs or specific integers.")
                 # Fill NaNs in numeric column, e.g., with 0 or mean/median if appropriate
                 self.df['lastupdated_numeric'] = self.df['lastupdated_numeric'].fillna(0) # Example: fill with 0
            print("Processed 'lastupdated' to numeric (seconds since epoch), with NaT handling.")
            self.df = self.df.drop(columns=['lastupdated_dt']) # Drop intermediate datetime column
        except Exception as e:
            print(f"Warning: Could not process 'lastupdated' column: {e}. It might be dropped or cause issues.")
            if 'lastupdated_numeric' not in self.df.columns: # If column creation failed entirely
                 self.df = self.df.drop(columns=['lastupdated'], errors='ignore')


        # Set Index
        self.df = self.df.set_index(['ticker', 'date']).sort_index()

        # Feature Selection
        potential_feature_columns = [col for col in self.df.columns if col not in ['ticker', 'date', 'lastupdated']]
        # Ensure only numeric features are selected
        self.feature_names = self.df[potential_feature_columns].select_dtypes(include=np.number).columns.tolist()
        
        # Explicitly add 'lastupdated_numeric' if it exists and is numeric, and not already there
        if 'lastupdated_numeric' in self.df.columns and pd.api.types.is_numeric_dtype(self.df['lastupdated_numeric']):
            if 'lastupdated_numeric' not in self.feature_names:
                self.feature_names.append('lastupdated_numeric')
        
        self.d_feat = len(self.feature_names)
        print(f"Selected features ({self.d_feat}): {self.feature_names}")


        # Label Calculation
        # Ensure 'closeadj' is numeric before transform
        if not pd.api.types.is_numeric_dtype(self.df['closeadj']):
            raise ValueError("Label column 'closeadj' is not numeric. Please check data.")
            
        self.df['label'] = self.df.groupby('ticker')['closeadj'].transform(
            lambda x: (x.shift(-1) - x) / x if x.notna().all() and len(x)>1 else np.nan # Added checks for robustness
        )

        # Data Segmentation
        self.features = []
        self.labels = []
        self.index_tuples = []

        grouped_by_ticker = self.df.groupby('ticker')
        for ticker_name, group_data in grouped_by_ticker:
            
            # Ensure features are numeric and handle NaNs before windowing
            # Using .loc to avoid SettingWithCopyWarning if group_data is a view
            group_features_df = group_data[self.feature_names].copy() 
            
            # Fill NaNs in features - using 0 as a placeholder strategy
            # A more robust strategy might involve forward-fill, mean imputation, or dropping rows/columns
            # if extensive NaNs are present that can't be meaningfully imputed.
            for col in self.feature_names:
                if group_features_df[col].isnull().any():
                    # print(f"Warning: NaN found in feature '{col}' for ticker '{ticker_name}'. Filling with 0.")
                    group_features_df[col] = group_features_df[col].fillna(0)

            feature_values = group_features_df.values
            labels_series = group_data['label']

            # Ensure enough data for at least one lookback window + 1 for the label
            if len(group_data) < self.lookback_window +1 : 
                # print(f"Skipping ticker {ticker_name}: not enough data for lookback window {self.lookback_window} and label.")
                continue

            for i in range(len(group_data) - self.lookback_window): 
                feature_sequence = feature_values[i : i + self.lookback_window]
                
                label_idx = i + self.lookback_window -1 
                label = labels_series.iloc[label_idx]
                
                current_date = group_data.index[label_idx][1] 

                if not np.isnan(label): 
                    self.features.append(feature_sequence)
                    self.labels.append(label)
                    self.index_tuples.append((current_date, ticker_name))
                # else:
                #     # This case is typically the last day(s) of a ticker's series where next day's data isn't available
                #     # print(f"Skipping sequence ending on {current_date} for {ticker_name} due to NaN label.")
                #     pass


    def get_index(self):
        if not self.index_tuples:
            return pd.MultiIndex(levels=[[], []], codes=[[], []], names=['datetime', 'instrument'])
        
        # Create MultiIndex from tuples
        multi_idx = pd.MultiIndex.from_tuples(self.index_tuples, names=['datetime', 'instrument'])
        
        # Sorting a MultiIndex can be done by converting to a DataFrame and back,
        # or by using sort_values on the MultiIndex itself if levels are already sorted.
        # For robustness, especially if tuples were not added in perfect order:
        if not multi_idx.is_monotonic_increasing:
             # This sorts based on the order of levels in names: 'datetime', then 'instrument'
            multi_idx = multi_idx.sort_values() 
        return multi_idx


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Ensure features are float32 and labels are float32 for PyTorch
        feature_sequence = np.array(self.features[idx], dtype=np.float32)
        label = np.array(self.labels[idx], dtype=np.float32)
        return feature_sequence, label

if __name__ == '__main__':
    csv_data_string = """ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated
T1,2023-01-01,10,11,9,10.5,1000,10.5,10.5,2023-01-01
T1,2023-01-02,10.5,11.5,10.2,11.0,1200,11.0,11.0,2023-01-02
T1,2023-01-03,11.0,11.2,10.8,11.1,800,11.1,11.1,03-01-2023
T1,2023-01-04,11.1,12.0,11.0,11.5,1500,11.5,11.5,2023-01-04
T1,2023-01-05,11.5,11.7,11.3,11.6,1100,11.6,11.6,2023-01-05
T1,2023-01-06,11.6,12.2,11.5,12.0,1300,12.0,12.0,2023-01-06
T1,2023-01-07,12.0,12.5,11.8,12.2,900,12.2,12.2,2023-01-07
T1,2023-01-08,12.2,12.8,12.1,12.5,1400,12.5,12.5,2023-01-08
T1,2023-01-09,12.5,13.0,12.4,12.8,1600,12.8,12.8,2023-01-09
T1,2023-01-10,12.8,13.5,12.7,13.0,1700,13.0,13.0,2023-01-10
T2,2023-01-01,100,101,99,100.5,2000,100.5,100.5,InvalidDate
T2,2023-01-02,100.5,102.5,100.2,101.0,2200,101.0,101.0,2023-01-02
T2,2023-01-03,101.0,101.5,100.8,101.2,1800,101.2,101.2,2023-01-03
T2,2023-01-04,101.2,103.0,101.0,102.0,2500,102.0,102.0,2023-01-04
T2,2023-01-05,102.0,102.5,101.5,102.3,2100,102.3,102.3,2023-01-05
T2,2023-01-06,102.3,103.5,102.1,103.0,2300,103.0,103.0,2023-01-06
T2,2023-01-07,103.0,104.0,102.8,103.5,1900,103.5,103.5,2023-01-07
T2,2023-01-08,103.5,105.0,103.2,104.0,2400,104.0,104.0,08/01/2023
T2,2023-01-09,104.0,106.0,103.8,105.0,2600,105.0,105.0,2023-01-09
T2,2023-01-10,105.0,107.0,104.5,106.0,2700,106.0,106.0,2023-01-10
"""
    csv_file_buffer = StringIO(csv_data_string)
    
    lookback = 8 # Adjusted lookback to ensure sequences can be generated from the sample data
    dataset = MultiIndexTimeSeriesDataset(csv_path_or_buffer=csv_file_buffer, lookback_window=lookback)

    print(f"\nNumber of features (d_feat): {dataset.d_feat}")
    print(f"Total number of sequences: {len(dataset)}")

    if len(dataset) > 0:
        # Test get_index()
        multi_index_df = dataset.get_index().to_frame() # Convert to DataFrame for easier inspection
        print("\nOutput of get_index().head(10):")
        print(multi_index_df.head(10))
        print("\nOutput of get_index().tail(10):")
        print(multi_index_df.tail(10))

        # Test __getitem__
        first_features, first_label = dataset[0]
        print(f"\nShape of the first feature sequence: {first_features.shape}, dtype: {first_features.dtype}")
        print(f"Label for the first sequence: {first_label}, dtype: {type(first_label)}")

        last_features, last_label = dataset[len(dataset)-1]
        print(f"\nShape of the last feature sequence: {last_features.shape}, dtype: {last_features.dtype}")
        print(f"Label for the last sequence: {last_label}, dtype: {type(last_label)}")
        
        # Verify sorting of get_index()
        print("\nIs get_index() sorted by datetime then instrument?")
        # Check if 'datetime' level is sorted
        datetime_sorted = multi_index_df['datetime'].is_monotonic_increasing
        # Check if 'instrument' is sorted within each group of 'datetime'
        # This is more complex to check directly, but sort_values in get_index should handle it.
        # A simpler check is if the whole index is sorted as per pandas definition.
        print(f"Index is monotonic increasing (overall check): {dataset.get_index().is_monotonic_increasing}")
        print(f"Datetime level is monotonic increasing: {datetime_sorted}")


    else:
        print("\nNo sequences were generated. Check data, lookback window, and processing steps (e.g., NaN handling, label calculation).")
