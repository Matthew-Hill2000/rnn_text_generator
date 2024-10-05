# RNN Text Generator

This project implements a Recurrent Neural Network (RNN) based text generator using PyTorch. It can be trained on any text dataset and can generate new text based on a given prefix.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Text Generation](#text-generation)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

## Project Structure

```
rnn_text_generator/
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── vocabulary.py
│   │   └── preprocessor.py
│   ├── models/
│   │   └── rnn.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       └── config_manager.py
│       └── prediction.py
│       └── tensorboard_utils.py
│
├── main.py  
│
├── configs/
│   └── default_config.json
│
├── text_data/
│   └── war_and_peace.txt
│
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Matthew-Hill2000/language_model.git
   cd rnn_text_generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Usage

The main script for training the model is `main.py`. You can run it with default settings or specify a custom configuration file:

```
python main.py --config configs/custom_config.json
```

## Configuration

The project uses a JSON configuration file to set various parameters. You can find the default configuration in `configs/default_config.json`. Here's an example of what it might look like:

```json
{
  "data": {
    "text_path": "Path to the input text file (default: 'data/input.txt')",
    "tokenization_mode": "Either 'char' for character-level or 'word' for word-level tokenization (default: 'char')",
    "num_steps": "Number of time steps (sequence length) for each training sample (default: 100)",
    "train_val_test_split": "Array of three float values representing the proportion of data for training, validation, and testing respectively (default: [0.8, 0.1, 0.1])"
    "preprocessors": [ 
            "lowercase",
            "remove_punctuation",
            "remove_extra_whitespace"
        ] "settings to decide how to preprocess the text before creating the corpus and vocabulary "
  },
  "model": {
    "type": "Type of the model",
    "hidden_dim": "Number of units in the hidden layers (default: 256)",
    "layer_dim": "Number of layers in the RNN (default: 2)",
    "dropout": "Dropout rate for regularization (float between 0 and 1, default: 0.2)",
    "pretrained_path": "Path to a pretrained model file (optional, default: null)"
  },
  "training": {
    "train": "Boolean indicating whether to train the model or use a pretrained one (default: true)",
    "batch_size": "Number of samples per batch of computation (default: 64)",
    "learning_rate": "Step size at each iteration while moving toward a minimum of the loss function (default: 0.001)",
    "num_epochs": "Number of complete passes through the training dataset (default: 10)",
    "save_model": "Boolean indicating whether to save the model after training (default: true)",
    "save_path": "Path where the trained model should be saved (default: 'models/trained_model.pth')",
  },
  "generation": {
    "generate": "Boolean indicating whether to generate text after training (default: False)",
    "prefix": "Starting text for generation (default: 'If they had known that you wished it')",
    "num_predictions": "Number of tokens to generate (default: 20)",
  },
  "logging": {
        "tensorboard_log_dir": "The directory within which to save the tensorboard logs"
}
```

You can create custom configuration files to experiment with different settings.

## Training

To train the model, run:

```
python main.py
```

This will use the default configuration. To use a custom configuration:

```
python main.py --config configs/custom_config.json
```

The training script will output the loss for each epoch and save the trained model.

## Text Generation

Text generation is performed automatically after training if `"generate": true` is set in the configuration file. You can specify the prefix and number of predictions in the configuration.

To generate text with a trained model without retraining:

1. Set `"train": false` in your configuration file.
2. Specify the path to your trained model in the configuration using the `"pretrained_path"` key under the `"model"` section.
3. Run the training script as usual.


