from abc import ABC
from abc import abstractmethod


class BaseModel(ABC):
    """ Abstract class for our models to extend. """

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def score(self, x, y):
        pass
