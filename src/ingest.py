import numpy as np
import pandas as pd
from pathlib import Path

# data ingest
base_dir = Path(__file__).resolve().parent.parent
filepath = base_dir / 'data' / 'pigeonflocks_trajectories' / 'ff1' / 'ff1_A.txt'
data = pd.read_csv(filepath, sep='\t', header=None, comment='#')
print(data.head())
