from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
import torch

class TensorboardWriter:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars to TensorBoard."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram of values to TensorBoard."""
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text_string: str, step: int):
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text_string, step)

    def log_hyperparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters and associated metrics to TensorBoard."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def add_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """Add a graph of the model to TensorBoard."""
        self.writer.add_graph(model, input_to_model)

    def close(self):
        """Close the SummaryWriter."""
        self.writer.close()