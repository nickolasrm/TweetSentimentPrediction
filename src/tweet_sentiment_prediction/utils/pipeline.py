import os
from pathlib import Path
from typing import Callable
from kedro.pipeline import Pipeline
from kedro.pipeline import node
import pandas as pd


def custom_pipeline(
    func: Callable,
    model: str,
    filepath: str = str(Path('data') / '05_model_input' / 'x_custom.csv')
) -> Pipeline:
    """Generates a pipeline with a node only if x_custom file exists

    Args:
        func (Callable)
        model (str): catalog model pickle name
        filepath (str, optional):
            Defaults to str(Path('data') / '05_model_input' / 'x_custom').

    Returns:
        Pipeline
    """
    if os.path.isfile(filepath):
        return Pipeline([
            node(
                func=func,
                inputs=[model, 'x_custom'],
                outputs='custom_prediction',
                name='test_custom_input'
            )
        ],
                        tags='custom')
    return Pipeline([])


def lowercase_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercases a DataFrame columns if dtype is object

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    for col in df:
        s = df[col]
        if pd.api.types.is_object_dtype(s):
            s = s.str.lower()
        df[col] = s
    return df
