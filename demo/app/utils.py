from functools import lru_cache
from pathlib import Path

import pandas as pd

from kirpandas import SemanticDataFrame

DATA_DIR = "./data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def load_dataframe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1251")
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";", encoding="cp1251")

    if df is None:
        raise UnicodeDecodeError

    return df


def get_dataframe_path(dataset_id: str) -> str:
    return f"{DATA_DIR}/{dataset_id}.csv"


def save_data(dataset_id: str, data: bytes) -> None:
    with open(get_dataframe_path(dataset_id), "wb") as f:
        f.write(data)


@lru_cache(maxsize=500)
def get_semantic_table(dataset_id: str):
    dataframe = load_dataframe(get_dataframe_path(dataset_id))
    return SemanticDataFrame(
        dataframe,
        columns=dataframe.columns,
    )


def train_semantic_table(
    semantic_dataframe: SemanticDataFrame, targets: list[str]
) -> None:
    semantic_dataframe.set_train_texts(pd.Series(targets))
    semantic_dataframe.fit_embedder()
    semantic_dataframe.fill_embeddings()
    _ = semantic_dataframe.embeddings_index
    _ = semantic_dataframe.table_index
