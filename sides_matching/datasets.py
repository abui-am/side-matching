import numpy as np
import pandas as pd
from wildlife_datasets import datasets

def _pop_reserved(kwargs):
    """Drop keys that wrappers set explicitly to avoid double-kwarg TypeErrors."""
    cleaned = dict(kwargs)
    cleaned.pop('img_load', None)
    return cleaned

def amvrakikos(root, transform=None, **kwargs):
    kwargs = _pop_reserved(kwargs)
    dataset = datasets.AmvrakikosTurtles(root, img_load='auto', transform=transform, **kwargs)
    dataset.df = modify_dates(dataset.df)
    return dataset

def reunion(root, transform=None, species=None, **kwargs):
    kwargs = _pop_reserved(kwargs)
    dataset = datasets.ReunionTurtles(root, img_load='auto', transform=transform, **kwargs)
    if species is not None:
        filtered = dataset.df[dataset.df['species'] == species]
        kwargs = {key: value for key, value in kwargs.items() if key != 'df'}
        dataset = datasets.ReunionTurtles(
            root, df=filtered, img_load='auto', transform=transform, **kwargs
        )
    dataset.df = modify_dates(dataset.df)
    return dataset

def reunion_green(root, **kwargs):
    return reunion(root, species='Green', **kwargs)

def reunion_hawksbill(root, **kwargs):
    return reunion(root, species='Hawksbill', **kwargs)

def zakynthos(root, transform=None, **kwargs):
    kwargs = _pop_reserved(kwargs)
    dataset = datasets.ZakynthosTurtles(root, img_load='auto', transform=transform, **kwargs)
    dataset.df = modify_dates(dataset.df)
    return dataset

def modify_dates(df):
    if 'date' in df.columns:
        date = df['date']
        idx = df.index[~date.isnull()]
        if len(idx) > 0 and len(str(date.iloc[idx[0]])) != 4:
            df['year'] = np.nan
            df.loc[idx, 'year'] = pd.to_datetime(df.loc[idx, 'date']).apply(lambda x: x.year)
        else:
            df['year'] = df['date']
    else:
        df['date'] = np.nan
        if 'year' not in df.columns:
            df['year'] = np.nan
    return df
