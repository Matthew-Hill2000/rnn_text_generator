import json
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "configs/training_config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}, creating default")
            
            config = self.create_default_config()
            self.validate_config(config)
            return config

        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.validate_config(config)
        return config

    def create_default_config(self) -> Dict[str, Any]:
        config = {
                    "data": {
                        "text_path": "text_data/war_and_peace.txt",
                        "tokenization_mode": "char",
                        "num_steps": 32,
                        "train_val_test_split": [
                            0.6,
                            0.2,
                            0.2
                        ],
                        "preprocessors": [
                            "lowercase",
                            "remove_punctuation",
                            "remove_extra_whitespace"
                        ]
                    },
                    "model": {
                        "type": "RNN",
                        "hidden_dim": 512,
                        "layer_dim": 1,
                        "dropout": 0.4,
                        "pretrained_path": "models/trained_model.pth"
                    },
                    "training": {
                        "train": True,
                        "batch_size": 512,
                        "learning_rate": 0.01,
                        "num_epochs": 7,
                        "save_model": True,
                        "save_path": "models/trained_model.pth"
                    },
                    "generation": {
                        "generate": True,
                        "prefix": "War of the Worlds",
                        "num_predictions": 50
                    },
                    "logging": {
                        "tensorboard_log_dir": "runs"
                    }
                }
        
        json_config = json.dumps(config, indent=4)
        with open("configs\default_config.json", "w") as file:
            file.write(json_config)

        return config

    def validate_config(self, config: Dict[str, Any]) -> None:
        required_keys = {
            'data': {'text_path', 'tokenization_mode', 'num_steps', 'train_val_test_split'},
            'model': {'type', 'hidden_dim', 'layer_dim'},
            'training': {'batch_size', 'learning_rate', 'num_epochs'}
        }

        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"Missing section in config: {section}")
            
            missing_keys = keys - set(config[section].keys())
            if missing_keys:
                raise ValueError(f"Missing keys in {section} section: {missing_keys}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.config.get(section, {}).get(key, default)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.config[key]