from abc import ABC, abstractmethod

import numpy as np


class EmbedderWrapper(ABC):
    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def get_word_vector(self, word: str) -> list | tuple | np.array: ...

    @abstractmethod
    def get_all_words(self) -> list | tuple | np.array: ...
