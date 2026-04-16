import numpy as np
import pandas as pd
from pathlib import Path

# data ingest
base_dir = Path(__file__).resolve().parent.parent
ff_files = sorted((base_dir / 'data' / 'pigeonflocks_trajectories').glob('ff*/ff*.txt'))
frames = []
for ff_file in ff_files:
    df = pd.read_csv(ff_file, sep='\t', header=None, comment='#')
    df["flight"] = ff_file.parent.name
    df['bird'] = ff_file.stem.split('_')[-1]
    df["source"] = str(ff_file)
    frames.append(df)

all_ff = pd.concat(frames, ignore_index=True)
print(all_ff.head())