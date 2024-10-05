from src.models.rnn import RNNModel
from src.data.dataset import TextDataset
from src.data.vocabulary import Vocab
from src.training.trainer import Trainer
import torch
from torch.nn import functional as F
import argparse
from src.utils.config_manager import ConfigManager
from src.utils.tensorboard_utils import TensorboardWriter

def predict(prefix: str, num_preds: int, vocab: 'Vocab', model: RNNModel, device: torch.device) -> str:
    """
    Generates predictions from a trained RNN model based on a given input prefix.

    Args:
        prefix (str): The input string to use as the starting point for predictions.
        num_preds (int): The number of prediction steps to generate.
        vocab (Vocab): The vocabulary object used to map tokens to indices and vice versa.
        model (RNNModel): The trained RNN model to use for making predictions.

    Returns:
        str: The generated sequence of predictions based on the input prefix.
    """
    if vocab.tokenization_mode == 'char':
        prefix_tokens = list(prefix.lower())
    else:  # word mode
        prefix_tokens = prefix.split(" ")

    state, outputs = None, [vocab[token] for token in prefix_tokens[:1]]

    for i in range(len(prefix_tokens) + num_preds - 1):
        X = torch.tensor([[outputs[-1]]])
        X_one_hot = F.one_hot(X.T, len(vocab)).type(torch.float32).to(device)

        rnn_outputs = model(X_one_hot)

        if i < len(prefix_tokens) - 1:  # Warm-up period
            outputs.append(vocab[prefix_tokens[i + 1]])
        else:  # Predict num_preds steps
            outputs.append(int(rnn_outputs.argmax(axis=2).reshape(1)))

    predicted_tokens = [vocab.idx_to_token[i] for i in outputs]

    if vocab.tokenization_mode == 'char':
        return ''.join(predicted_tokens)
    else:  # word mode
        return ' '.join(predicted_tokens)

def generate_samples(model, vocab, device, num_samples=5, prefix="The", num_tokens=50):
    samples = []
    for _ in range(num_samples):
        sample = predict(prefix, num_tokens, vocab, model, device)
        samples.append(sample)
    return samples