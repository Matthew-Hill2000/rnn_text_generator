import re
from typing import List, Callable

class Preprocessor:
    def __init__(self, steps: List[str]):
        self.steps = steps

    def process(self, text: str) -> str:
        for step in self.steps:
            if hasattr(self, step):
                text = getattr(self, step)(text)
            else:
                print(f"Warning: Preprocessing step '{step}' not found.")
        return text

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text: str) -> str:
        return re.sub(r'\d+', '', text)

    def remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

def create_preprocessor(steps: List[str]) -> Preprocessor:
    return Preprocessor(steps)