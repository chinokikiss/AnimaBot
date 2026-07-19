import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "tags_enhanced.csv")

def sample_tags(weighted_k=3, random_k=2, max_threshold=None, encoding='utf-8'):
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

    total_len = len(df_filtered)
    
    actual_weighted_k = min(weighted_k, total_len)
    actual_random_k = min(random_k, total_len - actual_weighted_k)

    sampled_indices_weighted = []
    sampled_indices_random = []

    if actual_weighted_k > 0:
        total_count = df_filtered['post_count'].sum()
        df_filtered['probability'] = df_filtered['post_count'] / total_count

        sampled_indices_weighted = np.random.choice(
            df_filtered.index, 
            size=actual_weighted_k, 
            replace=False, 
            p=df_filtered['probability']
        )

    if actual_random_k > 0:
        df_remaining = df_filtered.drop(index=sampled_indices_weighted)
        sampled_indices_random = np.random.choice(
            df_remaining.index, 
            size=actual_random_k, 
            replace=False
        )

    final_indices = list(sampled_indices_weighted) + list(sampled_indices_random)
    
    result = df_filtered.loc[final_indices, ['cn_name']].copy()
    return result

def replace_underscores(data):
    if isinstance(data, str):
        return data.replace('_', ' ')
    elif isinstance(data, list):
        return [replace_underscores(item) for item in data]
    elif isinstance(data, dict):
        return {replace_underscores(key): replace_underscores(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(replace_underscores(item) for item in data)
    else:
        return data