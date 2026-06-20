import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import pandas as pd

pd.read_parquet('market_data_cache_v3.parquet', engine='fastparquet')
print('Done!')
