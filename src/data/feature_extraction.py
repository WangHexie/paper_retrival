from typing import List

import pandas as pd


class UserFeature:
    def __init__(self, base_data: pd.DataFrame):
        self.base_data = base_data

    def get_features(self, user_id: List[str]):
        pass


class PaperFeature:
    def __init__(self, paper_data: pd.DataFrame):
        self.paper_data = paper_data

    def get_features(self, paper_id: List[str]):
        pass
