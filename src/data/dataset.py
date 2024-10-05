import re
import collections
from typing import List, Dict, Union, Tuple
import torch
from torch.utils.data import Dataset
from src.utils.config_manager import ConfigManager
from src.data.vocabulary import Vocab
from src.data.preprocessor import create_preprocessor

class TextDataset(Dataset):
    """
    A PyTorch dataset class for text data, providing sequences of tokens for training
    a recurrent neural network for language modeling.

    This class handles the loading, preprocessing, and tokenization of text data,
    as well as the creation of input-output pairs for training.

    Attributes:
        config (ConfigManager): The configuration manager instance.
        tokenization_mode (str): The mode of tokenization ('char' or 'word').
        num_steps (int): The number of tokens in each subsequence.
        text (str): The raw text data.
        corpus (torch.Tensor): The tokenized corpus as a tensor of indices.
        vocab (Vocab): The vocabulary object for the dataset.
        current_offset (int): The current offset for subsequence extraction.

    Methods:
        extract_text(path: str) -> str:
            Extracts the text from a file.
        
        preprocess(text: str) -> str:
            Preprocesses the raw text.
        
        tokenize(text: str) -> List[str]:
            Tokenizes the preprocessed text.
        
        build() -> Tuple[List[int], Vocab]:
            Builds the tokenized corpus and vocabulary.
        
        __getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            Retrieves a subsequence of tokens as an input-output pair.
        
        __len__() -> int:
            Returns the total number of subsequences in the dataset.
        
        set_offset(offset: int) -> None:
            Sets the current offset for subsequence extraction.
    """

    def __init__(self, config: ConfigManager):
        """Initializes the TextDataset instance."""
        self.config: ConfigManager = config
        self.tokenization_mode: str = config.get('data', 'tokenization_mode', 'char')
        self.num_steps: int = config.get('data', 'num_steps')
        
        text_path: str = config.get('data', 'text_path')
        self.text: str = self.extract_text(text_path)

        # Create preprocessor
        preprocessor_steps: List[str] = config.get('data', 'preprocessors', ['lowercase', 'remove_extra_whitespace'])
        self.preprocessor = create_preprocessor(preprocessor_steps)
        self.processed_text = self.preprocess(self.text)


        corpus, self.vocab = self.build()
        self.corpus: torch.Tensor = torch.tensor(corpus)
        self.current_offset: int = 0

    def extract_text(self, path: str) -> str:
        """Extracts the text from a file."""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def preprocess(self, text: str) -> str:
        """Preprocesses the raw text."""
        return self.preprocessor.process(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes the preprocessed text. """
        if self.tokenization_mode == 'char':
            return list(text)
        else:  # word-level
            return text.split()

    def build(self) -> Tuple[List[int], Vocab]:
        """Builds the tokenized corpus and vocabulary."""
        tokens = self.tokenize(self.processed_text)
        vocab = Vocab(tokens, min_freq=1, tokenization_mode=self.tokenization_mode)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a subsequence of tokens as an input-output pair."""
        start_idx = idx * self.num_steps + self.current_offset
        if start_idx >= len(self.corpus) - self.num_steps - 1:
            start_idx = start_idx % (len(self.corpus) - self.num_steps - 1)
        
        end_idx = start_idx + self.num_steps + 1
        sequence = self.corpus[start_idx:end_idx]
        
        return sequence[:-1], sequence[1:]

    def __len__(self) -> int:
        """Returns the total number of subsequences in the dataset."""
        return (len(self.corpus) - self.num_steps - 1) // self.num_steps

    def set_offset(self, offset: int) -> None:
        """Sets the current offset for subsequence extraction."""
        self.current_offset = offset