from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NerPredictor(ABC):
    model_data_path: str
    checkpoint_path: str
    output_path: str
    embedding_size: int
    hidden_nums: int
    learning_rate: float

    @abstractmethod
    def predict(self):
        ...

    @abstractmethod
    def output(self):
        ...

    def run(self):
        print("Start predicting...")
        self.predict()

        print("Start outputing")
        self.output()