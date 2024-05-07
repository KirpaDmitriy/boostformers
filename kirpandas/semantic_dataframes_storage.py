import faiss
import numpy as np

from .semantic_dataframe import SemanticDataFrame
from .wrappers import EmbedderWrapper


class SemanticDataFramesStorage:
    def __init__(self, embeddings_model: EmbedderWrapper):
        self._saved_ids = set()
        self._ids_sequence = []
        self._dataframes_index = faiss.IndexFlatL2(300)
        self._embeddings: EmbedderWrapper = embeddings_model

    @property
    def embeddings(self) -> EmbedderWrapper:
        if not self._embeddings.is_loaded:
            print("Загрузка эмбеддингов в хранилище наборов данных...")
            self._embeddings.load_model()
            print("Эмбеддинги в хранилище наборов данных загружены")
        return self._embeddings

    def add_semantic_dataframe(
        self, dataframe_id: str, semantic_dataframe: SemanticDataFrame
    ) -> None:
        if not isinstance(semantic_dataframe.description, str):
            raise ValueError(
                "Для сохранения набора данных в индексе, ему нужно задать описание `description`"
            )
        if dataframe_id in self._saved_ids:
            return
        self._ids_sequence.append(dataframe_id)
        self._saved_ids.add(dataframe_id)
        semantic_dataframe_embedding = self.embeddings.get_word_vector(
            semantic_dataframe.description
        )
        self._dataframes_index.add(np.array([semantic_dataframe_embedding]))

    def search_dataframes(self, text: str, n_results: int = 5) -> tuple[str]:
        text_embed = self.embeddings.get_word_vector(text)
        _, found_indexes = self._dataframes_index.search(
            np.array([text_embed]), n_results
        )
        return tuple(
            self._ids_sequence[found_index] for found_index in found_indexes[0]
        )
