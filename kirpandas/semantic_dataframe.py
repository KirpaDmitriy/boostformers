import functools
import os

import faiss
import fasttext
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from more_itertools import chunked


class SemanticDataFrame(pd.DataFrame):
    _metadata = pd.DataFrame._metadata + ["_description", "_embeddings_path"]

    def __init__(self, *args, **kwargs):
        description = kwargs.pop("description", None)
        embeddings_path = kwargs.pop("embeddings_path", None) or os.environ.get(
            "KIRPANDAS_EMBEDDINGS_PATH"
        )
        if not isinstance(embeddings_path, str):
            raise ValueError(
                "В виде строки необходимо передать путь к файлу с эмбеддингами в аргументе `embeddings_path`, либо в "
                "переменной среды окружения `KIRPANDAS_EMBEDDINGS_PATH`"
            )

        super().__init__(*args, **kwargs)

        if description is not None and not isinstance(description, str):
            raise ValueError("Параметр `description` должен быть строкой")
        self._description = description  # описание таблицы

        self._train_texts = (
            None  # тексты для обучения модели извлечения эмбеддингов строк таблицы
        )

        self._table_lines_embeddings = None  # подсчитанные эмбеды для строк таблицы

        self._embeddings_path = embeddings_path  # путь к модели эмбеддингов
        self._embeddings = None  # модель эмбеддингов
        self._embeddings_index = None  # индекс эмбеддингов

        self._text_model = None  # модель извлечения эмбеддингов строк таблицы

        self._table_index = None  # индекс эмбеддингов строк таблицы

    @property
    def _constructor(self):
        custom_attributes = {}
        for attr in self._metadata:
            if attr not in pd.DataFrame._metadata:
                custom_attributes[attr.removeprefix("_")] = getattr(self, attr)
        return functools.partial(self.__class__, **custom_attributes)

    def set_description(self, description: str) -> None:
        if not isinstance(description, str):
            raise ValueError("Параметр `description` должен быть строкой")
        self._description = description

    @property
    def description(self) -> str | None:
        return self._description

    def set_train_texts(self, train_texts: pd.Series) -> None:
        self._train_texts = train_texts

    @property
    def train_texts(self) -> pd.Series:
        return self._train_texts

    def choose_model(self): ...

    @property
    def embeddings(self):
        if not self._embeddings:
            print("Загрузка эмбеддингов...")
            self._embeddings = fasttext.load_model(
                self._embeddings_path
            )  # модель эмбеддингов
            print("Эмбеддинги загружены")
        return self._embeddings

    @property
    def embeddings_index(self):
        if not self._embeddings_index:
            print("Загрузка индекса эмбеддингов...")
            self._embeddings_index = faiss.IndexFlatL2(300)
            for i, batch in enumerate(chunked(self._embeddings.get_words(), 100_000)):
                self._embeddings_index.add(
                    np.array([self._embeddings.get_word_vector(word) for word in batch])
                )
            print(
                f"Индекс эмбеддингов загружен. Количество элементов: {self._embeddings_index.ntotal}"
            )
        return self._embeddings_index

    @property
    def table_index(self):
        if not self._table_index:
            print("Загрузка индекса строк таблиц...")
            self._table_index = faiss.IndexFlatL2(300)
            for batch in chunked(self._table_lines_embeddings, 100_000):
                self._table_index.add(np.array(batch))
            print(
                f"Индекс строк таблиц загружен. Количество элементов: {self._table_index.ntotal}"
            )
        return self._table_index

    def fit_embedder(self):
        text_indexes = self._train_texts.notna() & (self._train_texts.str.len() > 0)
        train_features = self[text_indexes]
        train_texts = self._train_texts[text_indexes]
        train_embeddings = np.array(
            [self.embeddings.get_word_vector(train_text) for train_text in train_texts]
        )
        self._text_model = CatBoostRegressor(
            iterations=100,
            depth=int(np.sqrt(len(self.columns))) + 1,
            learning_rate=0.1,
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
        )

        self._text_model.fit(train_features, train_embeddings)

        return self._text_model

    def fill_embeddings(
        self,
    ) -> None:  # оптимизировать вычисление эмбеддов только для уже невычисленных
        self._table_lines_embeddings = self._text_model.predict(self)

    def generate_texts(self, lines: np.array) -> list:
        final_answer = []
        embeds = self._text_model.predict(lines)
        _, found_indexes = self.embeddings_index.search(embeds, 5)
        for answer in found_indexes:
            final_answer.append([])
            for sub_answer in answer:
                final_answer[-1].append(self.embeddings.get_words()[sub_answer])
        return final_answer

    def search_lines(self, text: str, n_lines: int = 5) -> pd.DataFrame:
        _, found_indexes = self.table_index.search(
            np.array([self.embeddings.get_word_vector(text)]), n_lines
        )
        return self.iloc[found_indexes[0]]
