import os

import fasttext
import numpy as np

from kirpandas.wrappers import EmbedderWrapper


class FastTextEmbedder(EmbedderWrapper):
    def __init__(self, embeddings_path: str | None = None):
        embeddings_path = embeddings_path or os.environ.get("KIRPANDAS_EMBEDDINGS_PATH")
        if not isinstance(embeddings_path, str):
            raise ValueError(
                "В виде строки необходимо передать путь к файлу с эмбеддингами в аргументе `embeddings_path`, либо в "
                "переменной среды окружения `KIRPANDAS_EMBEDDINGS_PATH`"
            )

        self.embeddings_path = embeddings_path
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return bool(self._model)

    def load_model(self) -> None:
        self._model = fasttext.load_model(self.embeddings_path)

    def get_word_vector(self, word: str) -> np.array:
        return self._model.get_word_vector(word)

    def get_all_words(self) -> np.array:
        return self._model.get_words()


fasttext_embedder = FastTextEmbedder()
