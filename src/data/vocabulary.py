import re
import collections
from typing import List, Dict, Union, Tuple
import torch
from torch.utils.data import Dataset
from src.utils.config_manager import ConfigManager

class Vocab:
    """
    A class for building and managing a vocabulary for language models.

    This class constructs a vocabulary from a given list of tokens, providing methods
    to convert between tokens and their corresponding indices, as well as handling
    out-of-vocabulary tokens.

    Attributes:
        tokenization_mode (str): The mode of tokenization ('char' or 'word').
        token_freqs (List[Tuple[str, int]]): A sorted list of (token, frequency) pairs.
        idx_to_token (List[str]): A list mapping indices to tokens.
        token_to_idx (Dict[str, int]): A dictionary mapping tokens to their indices.

    Methods:
        __getitem__(tokens: Union[str, List[str]]) -> Union[int, List[int]]:
            Converts token(s) to their corresponding index/indices.
        
        to_tokens(indices: Union[int, List[int]]) -> Union[str, List[str]]:
            Converts index/indices back to their corresponding token(s).
        
        unk() -> int:
            Returns the index of the unknown token.
        
        __len__() -> int:
            Returns the total number of tokens in the vocabulary.
    """

    def __init__(self, tokens: List[str] = [], min_freq: int = 0, 
                 reserved_tokens: List[str] = [], tokenization_mode: str = 'char'):
        """
        Initializes the Vocab instance.

        Args:
            tokens (List[str]): A list of tokens to build the vocabulary from.
            min_freq (int): Minimum frequency for a token to be included in the vocabulary.
            reserved_tokens (List[str]): Tokens to always include in the vocabulary.
            tokenization_mode (str): The mode of tokenization ('char' or 'word').
        """
        self.tokenization_mode: str = tokenization_mode
        
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        
        counter = collections.Counter(tokens)
        self.token_freqs: List[Tuple[str, int]] = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        self.idx_to_token: List[str] = ['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq
        ]
        self.token_to_idx: Dict[str, int] = {token: idx for idx, token in enumerate(self.idx_to_token)}
    
    def __len__(self) -> int:
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]
    
    @property
    def unk(self) -> int:
        return self.token_to_idx['<unk>']