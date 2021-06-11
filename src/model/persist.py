import os
import pickle
from pathlib import Path
from ..config import root_path


class PersistModel:
    def __init__(self, name):
        self.name = name
        self.path = Path(root_path, "cache", self.name)

    def exist(self):
        return os.path.exists(self.path)

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        return data

    def save(self, data):
        with open(self.path, "wb") as f:
            pickle.dump(data, f)
        return self
