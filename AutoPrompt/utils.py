import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "tags_enhanced.csv")

def sample_tags(top_k=5, max_threshold=None, encoding='utf-8'):
    try:
        df = pd.read_csv(CSV_PATH, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding='gb18030')

    df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)
    
    if max_threshold is not None:
        df_filtered = df[(df['post_count'] > 0) & (df['post_count'] <= max_threshold)].copy()
    else:
        df_filtered = df[df['post_count'] > 0].copy()
    
    if df_filtered.empty:
        print("没有找到符合条件的有效 post_count 数据进行采样。")
        return pd.DataFrame()

    total_count = df_filtered['post_count'].sum()
    df_filtered['probability'] = df_filtered['post_count'] / total_count

    sample_size = min(top_k, len(df_filtered))

    sampled_indices = np.random.choice(
        df_filtered.index, 
        size=sample_size, 
        replace=False, 
        p=df_filtered['probability']
    )

    result = df_filtered.loc[sampled_indices, ['cn_name']]
    return result