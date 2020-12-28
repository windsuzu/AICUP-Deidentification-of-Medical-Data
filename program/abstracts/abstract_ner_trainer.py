from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NerTrainer(ABC):
    train_data_path: str
    model_data_path: str
    checkpoint_path: str
    checkpoint_keep: int
    max_sentence_length: int
    batch_size: int
    embedding_size: int
    hidden_nums: int
    epochs: int
    learning_rate: float
    isVisualize: bool

    @abstractmethod
    def tokenize(self):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def visualize(self):
        ...

    def run(self):
        print("Start tokenization...")
        self.tokenize()
        print("Start training...")
        self.train()
        
        if self.isVisualize:
            self.visualize()