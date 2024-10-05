import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from src.data.dataset import TextDataset
from src.utils.config_manager import ConfigManager

class RNNModel(nn.Module):
    """
    A recurrent neural network (RNN) model using PyTorch's `nn.RNN` module to process sequences of tokens and predict
    outputs for each time step.

    Args:
        vocab_size (int): The size of the input vocabulary. Each token in the input sequence is expected to be
        represented as a one-hot vector of size `vocab_size`.
        hidden_dim (int): The number of features in the hidden state, i.e., the dimensionality of the RNN's hidden
        layer.
        layer_dim (int): The number of recurrent layers. A value of 1 indicates a simple RNN, while higher values
        indicate stacked RNNs (deep RNN).
        output_dim (int): The size of the output vector for each time step, typically equal to the vocabulary size in
        language modeling tasks.

    Attributes:
        hidden_dim (int): Number of features in the hidden state of the RNN.
        layer_dim (int): Number of recurrent layers in the RNN.
        rnn (nn.RNN): The RNN layer that processes the input sequence, producing hidden states at each time step.
        fc (nn.Linear): A fully connected (linear) layer that maps the hidden state at each time step to the output
        dimension.

    Methods:
        forward(x):
            Defines the forward pass through the network. Takes an input tensor `x`, initializes the hidden state,
            passes `x` through the RNN, and applies the fully connected layer to the output of the RNN at each time
            step.

            Args:
                x (torch.Tensor): Input tensor of shape (num_time_steps, batch_size, vocab_size), where `num_time_steps`
                is the length of the input sequence, `batch_size` is the number of sequences in the batch, and
                `vocab_size` is the size of the one-hot encoded input vector for each token.

            Returns:
                torch.Tensor: Output tensor of shape (num_time_steps, batch_size, output_dim), where `output_dim` is the
                size of the output vector for each time step (e.g., equal to the vocabulary size in language modeling
                tasks).
    """
    def __init__(self, config: ConfigManager, vocab_size: int):
        super(RNNModel, self).__init__()
        
        self.hidden_dim = config.get('model', 'hidden_dim')
        self.layer_dim = config.get('model', 'layer_dim')
        self.dropout = config.get('model', 'dropout', 0.0)
        
        self.rnn = nn.LSTM(
            vocab_size, 
            self.hidden_dim, 
            self.layer_dim, 
            batch_first=False,
            dropout=self.dropout if self.layer_dim > 1 else 0
        )

        self.fc = nn.Linear(self.hidden_dim, vocab_size)
    
    def forward(self, x):
        # x is expected to have shape (num_time_steps, batch_size, vocab_size)

        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)


        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = torch.stack([self.fc(out[i,:,:]) for i in range(out.shape[0])], 0)       
        
        return out
    
        