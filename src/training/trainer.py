import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from src.data.dataset import TextDataset
from src.models.rnn import RNNModel
from src.utils.config_manager import ConfigManager
from src.utils.tensorboard_utils import TensorboardWriter

class Trainer:
    """
    A class that handles the training and evaluation process for an RNN model using a dataset. 
    It manages data loading, model instantiation, and the training loop for multiple epochs.

    Args:
        model_class (nn.Module): The neural network class to instantiate for training.
        model_kwargs (dict): The keyword arguments required to instantiate the model class.
        dataset (Dataset): The dataset to use for training, validation, and testing.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for data loading.
        train_val_test_split (tuple): A tuple containing three values that represent the proportion 
                                      of the dataset to use for training, validation, and testing, respectively. For
                                      example, (0.8, 0.1, 0.1).

    Attributes:
        device (torch.device): The device ('cuda' or 'cpu') on which to train the model.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for data loading.
        train_val_test_split (tuple): Proportion of the dataset used for training, validation, and testing.
        criterion (nn.CrossEntropyLoss): Loss function used during training.
        optimizer (optim.Adam): Optimizer for updating model parameters.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set (if used).
        model_class (nn.Module): Neural network class to be instantiated.
        model_kwargs (dict): Arguments to be passed to the model class.
        model (nn.Module): The instantiated neural network model.
        dataset (Dataset): The dataset used for training, validation, and testing.

    Methods:
        load_data():
            Splits the dataset into training, validation, and test sets, and creates DataLoader objects
            for each subset.

        load_model():
            Initializes the model, loss function, and optimizer. Loads the model onto the selected device (CPU or GPU).

        train_one_epoch():
            Trains the model for one epoch, iterating over the training set. Computes the loss and 
            updates the model parameters via backpropagation.

        evaluate():
            Evaluates the model's performance on the validation set. Computes the average loss over 
            the validation set without updating the model parameters.

        train():
            Trains the model for the specified number of epochs. Calls `train_one_epoch` for training 
            and `evaluate` for validation after each epoch, and prints the training and validation loss.

    """
    def __init__(self,config: ConfigManager, model_class, dataset):

        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = config.get('training', 'learning_rate', 0.001)
        self.num_epochs = config.get('training', 'num_epochs', 10)
        self.batch_size = config.get('training', 'batch_size', 64)
        
        split = config.get('data', 'train_val_test_split')
        self.train_val_test_split = tuple(split)
        
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

        self.model_class = model_class
        self.model = None
        self.dataset = dataset

        self.vocab_size = len(self.dataset.vocab)

        log_dir = config.get('logging', 'tensorboard_log_dir', 'runs')
        self.tb_writer = TensorboardWriter(log_dir)

        self.print_hyperparameters(self.config)

    def print_hyperparameters(self, config: ConfigManager) -> None:
        """
        Print the important hyperparameters from the configuration.
        
        Args:
            config (ConfigManager): The configuration manager object.
        """
        print("=== Hyperparameters ===")
        print(f"Text path: {config.get('data', 'text_path')}")
        print(f"Tokenization mode: {config.get('data', 'tokenization_mode')}")
        print(f"Sequence length: {config.get('data', 'num_steps')}")
        print(f"Model type: {config.get('model', 'type')}")
        print(f"Hidden dimensions: {config.get('model', 'hidden_dim')}")
        print(f"Number of layers: {config.get('model', 'layer_dim')}")
        print(f"Dropout: {config.get('model', 'dropout', 0.0)}")
        print(f"Batch size: {config.get('training', 'batch_size')}")
        print(f"Learning rate: {config.get('training', 'learning_rate')}")
        print(f"Number of epochs: {config.get('training', 'num_epochs')}")
        print("=======================")        

    def load_data(self):
        
        num_train = int(len(self.dataset) * self.train_val_test_split[0])
        num_val = int(len(self.dataset) * self.train_val_test_split[1])
        num_test = len(self.dataset) - num_train - num_val

        train_dataset = Subset(self.dataset, list(range(0, num_train)))
        val_dataset = Subset(self.dataset, list(range(num_train, num_train + num_val)))
        test_dataset = Subset(self.dataset, list(range(num_train + num_val, len(self.dataset))))
        
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size)

    def load_model(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {self.device}")

        self.model = self.model_class(self.config, self.vocab_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        is_on_gpu = next(self.model.parameters()).is_cuda
        print(f"Model is on GPU:{is_on_gpu}")

        self.log_model_graph()

    def log_model_graph(self):
        # Create a sample input tensor
        sample_input = torch.zeros((self.config.get('data', 'num_steps'), 1, self.vocab_size), dtype=torch.float32).to(self.device)
        
        # Log the model graph
        self.tb_writer.add_graph(self.model, sample_input)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to the specified path.

        Args:
            path (str): The path where the model should be saved.
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train_one_epoch(self):
        """Train the model for one epoch."""
        self.model.train()  
        epoch_loss = 0

        new_offset = torch.randint(0, self.dataset.num_steps, (1,)).item()
        self.dataset.set_offset(new_offset)
        
        for idx, (X, Y) in enumerate(self.train_loader):
            X, Y = X.to(self.device), Y.to(self.device)  

            
            vocab_size = len(self.dataset.vocab)  
            X_one_hot = F.one_hot(X.T, vocab_size).type(torch.float32).to(self.device)
            Y_one_hot = F.one_hot(Y.T, vocab_size).type(torch.float32).to(self.device)

            outputs = self.model(X_one_hot).permute(1, 0, 2)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            labels = Y_one_hot.permute(1, 0, 2).reshape(-1, Y_one_hot.shape[-1])

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            epoch_loss += loss.item()
            
            # Log batch loss to TensorBoard
            global_step = (self.current_epoch - 1) * len(self.train_loader) + idx
            self.tb_writer.log_scalar('Batch/Train Loss', loss.item(), global_step)

        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()  
        epoch_loss = 0
        
        with torch.no_grad():
            for idx, (X, Y) in enumerate(self.val_loader):
                X, Y = X.to(self.device), Y.to(self.device)

                vocab_size = self.model.fc.out_features
                X_one_hot = F.one_hot(X.T, vocab_size).type(torch.float32).to(self.device)
                Y_one_hot = F.one_hot(Y.T, vocab_size).type(torch.float32).to(self.device)

                outputs = self.model(X_one_hot).permute(1, 0, 2)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = Y_one_hot.permute(1, 0, 2).reshape(-1, Y_one_hot.shape[-1])

                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                # print(f"batch {idx} / {len(self.train_loader)}")
        return epoch_loss / len(self.val_loader)

    def train(self):
        """Train the model for the given number of epochs."""
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

             # Log epoch losses to TensorBoard
            self.tb_writer.log_scalars('Epoch Losses', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)

            # Log model parameters histograms
            for name, param in self.model.named_parameters():
                self.tb_writer.log_histogram(f'Parameters/{name}', param.data, epoch + 1)
                if param.grad is not None:
                    self.tb_writer.log_histogram(f'Gradients/{name}', param.grad, epoch + 1)


        # Save the model if specified in the config
        if self.config.get('training', 'save_model', False):
            save_path = self.config.get('training', 'save_path', 'models/trained_model.pth')
            self.save_model(save_path)


        # Log hyperparameters and final metrics
        hparam_dict = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'hidden_dim': self.config.get('model', 'hidden_dim'),
            'layer_dim': self.config.get('model', 'layer_dim'),
        }
        metric_dict = {'final_train_loss': train_loss, 'final_val_loss': val_loss}
        self.tb_writer.log_hyperparams(hparam_dict, metric_dict)

        # Close the TensorBoard writer
        self.tb_writer.close()


