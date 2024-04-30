import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from kirpandas import SemanticDataFrame

DATA_DIR = os.environ.get("KIRPANDAS_DEMO_APP_STORAGE") or "./data"
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


def get_dataframe_path(dataframe_id: str) -> str:
    return f"{DATA_DIR}/{dataframe_id}.csv"


def get_dataframe_index_path(dataframe_id: str) -> str:
    return f"{DATA_DIR}/{dataframe_id}_index.csv"


def get_embeddings_index_path() -> str:
    return f"{DATA_DIR}/embeddings_index.csv"


def save_data(dataframe_id: str, data: bytes) -> None:
    with open(get_dataframe_path(dataframe_id), "wb") as f:
        f.write(data)


@lru_cache(maxsize=500)
def get_semantic_table(dataframe_id: str):
    dataframe = load_dataframe(get_dataframe_path(dataframe_id))
    kwargs = {}
    dataframe_index_path = get_dataframe_index_path(dataframe_id)
    embeddings_index_path = get_embeddings_index_path()
    if os.path.exists(dataframe_index_path):
        kwargs["dataframe_index_path"] = dataframe_index_path
    if os.path.exists(embeddings_index_path):
        kwargs["embeddings_index_path"] = embeddings_index_path
    print(dataframe.to_json(orient="columns"))
    return SemanticDataFrame(dataframe, columns=dataframe.columns, **kwargs)


def train_semantic_table(
    semantic_dataframe: SemanticDataFrame, targets: list[str]
) -> None:
    semantic_dataframe.set_train_texts(pd.Series(targets))
    semantic_dataframe.set_embeddings_extraction_model(
        CatBoostRegressor(
            iterations=100,
            depth=int(np.sqrt(len(semantic_dataframe.columns))) + 1,
            learning_rate=0.1,
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
        )
    )
    semantic_dataframe.fit_embeddings_extraction_model()
    semantic_dataframe.load_embeddings_index()
    semantic_dataframe.load_dataframe_index()
