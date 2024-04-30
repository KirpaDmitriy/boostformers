import functools
import os
from typing import Any

import faiss
import fasttext
import numpy as np
import pandas as pd
from more_itertools import chunked


class SemanticDataFrame(pd.DataFrame):
    _metadata = pd.DataFrame._metadata + [
        "_description",
        "_embeddings_path",
        "_embeddings_index_path",
        "_dataframe_index_path",
    ]

    def __init__(self, *args, **kwargs):
        description = kwargs.pop("description", None)

        embeddings_path = kwargs.pop("embeddings_path", None) or os.environ.get(
            "KIRPANDAS_EMBEDDINGS_PATH"
        )
        if embeddings_path is not None and not isinstance(embeddings_path, str):
            raise ValueError(
                "В виде строки необходимо передать путь к файлу с эмбеддингами в аргументе `embeddings_path`, либо в "
                "переменной среды окружения `KIRPANDAS_EMBEDDINGS_PATH`"
            )

        embeddings_index_path = kwargs.pop(
            "embeddings_index_path", None
        ) or os.environ.get("KIRPANDAS_EMBEDDINGS_INDEX_PATH")

        dataframe_index_path = kwargs.pop(
            "dataframe_index_path", None
        ) or os.environ.get("KIRPANDAS_DATAFRAME_INDEX_PATH")

        super().__init__(*args, **kwargs)

        if description is not None and not isinstance(description, str):
            raise ValueError("Параметр `description` должен быть строкой")
        self._description = description  # описание таблицы

        self._embeddings_path = embeddings_path  # путь к модели эмбеддингов
        self._embeddings = None  # модель эмбеддингов
        self._embeddings_index = None  # индекс всего эмбеддингового пространства
        self._embeddings_index_path = (
            embeddings_index_path  # путь к индексу всего эмбеддингового пространства
        )

        self._train_texts = (
            None  # тексты для обучения извлечения эмбеддингов строк таблицы
        )
        self._embeddings_extraction_model = (
            None  # модель извлечения эмбеддингов строк таблицы
        )
        self._dataframe_lines_embeddings = None  # подсчитанные эмбеды для строк таблицы
        self._dataframe_index = None  # индекс эмбеддингов строк таблицы
        self._dataframe_index_path = (
            dataframe_index_path  # путь к индексу эмбеддингов строк таблицы
        )

    @property
    def _constructor(self):
        custom_attributes = {}
        for attr in self._metadata:
            if attr not in pd.DataFrame._metadata:
                custom_attributes[attr.removeprefix("_")] = getattr(self, attr)
        return functools.partial(self.__class__, **custom_attributes)

    ##### БЛОК ОПИСАНИЯ #####

    def set_description(self, description: str) -> None:
        if not isinstance(description, str):
            raise ValueError("Параметр `description` должен быть строкой")
        self._description = description

    @property
    def description(self) -> str | None:
        return self._description

    ##### БЛОК ЭМБЕДДИНГОВ ВСЕГО ПРОСТРАНСТВА #####

    @property
    def embeddings(self):
        if not self._embeddings:
            print("Загрузка эмбеддингов...")
            self._embeddings = fasttext.load_model(
                self._embeddings_path
            )  # модель эмбеддингов
            print("Эмбеддинги загружены")
        return self._embeddings

    def set_embeddings_index(self, path: str) -> None:
        self._embeddings_index = faiss.read_index(path)

    def load_embeddings_index(self) -> None:
        if not self._embeddings_index:
            if self._embeddings_index_path:
                print("Загрузка предпосчитанного индекса эмбеддингов с диска...")
                self.set_embeddings_index(self._embeddings_index_path)
                print(
                    f"Индекс эмбеддингов загружен. Количество элементов: {self._embeddings_index.ntotal}"
                )
            else:
                print("Загрузка индекса эмбеддингов...")
                self._embeddings_index = faiss.IndexFlatL2(300)
                for i, batch in enumerate(
                    chunked(self._embeddings.get_words(), 100_000)
                ):
                    self._embeddings_index.add(
                        np.array(
                            [self._embeddings.get_word_vector(word) for word in batch]
                        )
                    )
                print(
                    f"Индекс эмбеддингов загружен. Количество элементов: {self._embeddings_index.ntotal}"
                )

    @property
    def embeddings_index(self):
        self.load_embeddings_index()
        return self._embeddings_index

    def dump_embeddings_index(self, path: str) -> None:
        faiss.write_index(self.embeddings_index, path)

    ##### БЛОК ТРЕНИРОВОВЧНЫХ ТЕКСТОВ #####

    def set_train_texts(self, train_texts: pd.Series) -> None:
        self._train_texts = train_texts

    @property
    def train_texts(self) -> pd.Series:
        return self._train_texts

    ##### БЛОК МОДЕЛИ ИЗВЛЕЧЕНИЯ ЭМБЕДДИНГОВ #####

    def set_embeddings_extraction_model(self, model: Any) -> None:
        if (
            hasattr(model, "fit")
            and callable(getattr(model, "fit"))
            and hasattr(model, "predict")
            and callable(getattr(model, "predict"))
        ):
            self._embeddings_extraction_model = model
        else:
            raise ValueError(
                f"Модель {model} типа {type(model)} не имеет одного из методов: `fit` или `predict`"
            )

    @property
    def embeddings_extraction_model(self) -> Any | None:
        return self._embeddings_extraction_model

    def fit_embeddings_extraction_model(self):
        train_text_indexes = self._train_texts.notna() & (
            self._train_texts.str.len() > 0
        )
        print(train_text_indexes)
        print(train_text_indexes.shape)
        train_features = self[train_text_indexes]
        train_texts = self._train_texts[train_text_indexes]
        print("Getting word vectors for model train...")
        _ = self.embeddings
        print("Getting word vectors for finished. Forming vectors")
        train_embeddings = np.array(
            [self.embeddings.get_word_vector(train_text) for train_text in train_texts]
        )
        print("Word vectors for model train formed. Training model...")

        self.embeddings_extraction_model.fit(train_features, train_embeddings)
        print("Model trained")

        return self.embeddings_extraction_model

    ##### БЛОК ЭМБЕДДИНГОВ ТАБЛИЦЫ #####

    def fill_dataframe_embeddings(self) -> None:
        self._dataframe_lines_embeddings = self.embeddings_extraction_model.predict(
            self
        )

    @property
    def dataframe_lines_embeddings(self) -> np.array:
        if not self._dataframe_lines_embeddings:
            self.fill_dataframe_embeddings()
        return self._dataframe_lines_embeddings

    def set_dataframe_index(self, path: str) -> None:
        self._dataframe_index = faiss.read_index(path)

    def load_dataframe_index(self) -> None:
        if not self._dataframe_index:
            if self._dataframe_index_path:
                print("Загрузка предпосчитанного индекса строк таблиц с диска...")
                self.set_dataframe_index(self._dataframe_index_path)
                print(
                    f"Индекс строк таблиц загружен. Количество элементов: {self._dataframe_index.ntotal}"
                )
            else:
                print("Загрузка индекса строк таблиц...")
                self._dataframe_index = faiss.IndexFlatL2(300)
                for batch in chunked(self.dataframe_lines_embeddings, 100_000):
                    self._dataframe_index.add(np.array(batch))
                print(
                    f"Индекс строк таблиц загружен. Количество элементов: {self._dataframe_index.ntotal}"
                )

    @property
    def dataframe_index(self):
        self.load_dataframe_index()
        return self._dataframe_index

    def dump_dataframe_index(self, path: str) -> None:
        faiss.write_index(self.dataframe_index, path)

    ##### БЛОК ПРИКЛАДНЫХ МЕТОДОВ #####

    def search_lines(self, text: str, n_lines: int = 5) -> pd.DataFrame:
        _, found_indexes = self.dataframe_index.search(
            np.array([self.embeddings.get_word_vector(text)]), n_lines
        )
        return self.iloc[found_indexes[0]]

    def generate_texts(self, lines: np.array) -> list:
        final_answer = []
        embeds = self.embeddings_extraction_model.predict(lines)
        _, found_indexes = self.embeddings_index.search(embeds, 5)
        for answer in found_indexes:
            final_answer.append([])
            for sub_answer in answer:
                final_answer[-1].append(self.embeddings.get_words()[sub_answer])
        return final_answer
