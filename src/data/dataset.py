import os
import pandas as pd


class Dataset:
    base_path = "dataset"

    def _read_jsons(self, relative_path):
        path_to_json = os.path.join(self.base_path, relative_path)
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        jsons = [pd.read_json(os.path.join(path_to_json, file_name)) for file_name in json_files]
        return pd.concat(jsons, ignore_index=True)

    def read_base_dataset(self):
        path = "base_data"
        return self._read_jsons(path)

    def read_train_dataset(self):
        path = "train"
        return pd.read_json(os.path.join(self.base_path, path, "publications.json")), pd.read_json(
            os.path.join(self.base_path, path, "results.json"))

    def read_valid_dataset(self):
        path = "valid"
        return pd.read_json(os.path.join(self.base_path, path, "publications.json"))


if __name__ == '__main__':
    Dataset().read_valid_dataset()
    Dataset().read_train_dataset()
    Dataset().read_base_dataset()
