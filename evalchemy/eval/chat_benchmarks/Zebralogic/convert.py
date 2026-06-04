import pandas as pd
import json
from datasets import load_dataset

def format_solution(solution):
    header = solution['header']
    rows = solution['rows']
    parts = []
    for row in rows:
        part = ', '.join([f"{h}: {v}" for h, v in zip(header, row)])
        parts.append(part)
    return '\n'.join(parts)  # newline instead of ' | '
    
def reformat_zebralogic(df):
    df['input'] = df['puzzle']
    df['target'] = df['solution'].apply(format_solution)
    return df[['input', 'target']]

ds = load_dataset('WildEval/ZebraLogic', 'grid_mode', split='test')
df = ds.to_pandas()
reformatted_df = reformat_zebralogic(df)
print(f"Total examples: {len(reformatted_df)}")
print(reformatted_df.head())
reformatted_df.to_json('zebralogic_reformatted.json', orient='records', lines=True)