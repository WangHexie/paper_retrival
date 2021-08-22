import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

from src.application.retrieve_for_evaluation import Retrieve
from src.data.data_transform import paper_data_transformation, base_data_transformation
from src.model.retrieval import BiEncoderRetrieval


class BiEncoderRetrievalTrain:
    def __init__(self, model_name, device="cuda:0", loss="triplet",hard_neg=True):
        self.loss = loss
        self.hard_neg = hard_neg
        if hard_neg:
            self.loss = "MultipleNegativesRankingLoss"
        retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=60, save=False),
                                   transformation_kwargs=dict(add_title=True))
        self.retrieval_model = retrieval_model

        self.model = BiEncoderRetrieval(model_name=model_name, batch_size=32, device=device, loss=self.loss)

        self.pubs = retrieval_model.pubs
        self.user_info = retrieval_model.base_data
        self.labels = retrieval_model.labels

        self._setup()

    def _setup(self):
        pubs_string = paper_data_transformation(self.pubs, add_abstract=True, add_title=True, abstract_length=200)
        user_string = base_data_transformation(self.user_info, log_transform=True)

        # warning: data reshape????
        self.user_string = pd.DataFrame(data=user_string.values, index=self.user_info["id"])
        self.pubs_string = pd.DataFrame(data=pubs_string.values, index=self.pubs["id"])

        return self

    # def _train_model(self):

    def retrieve_hard_negative(self):
        user_predictions = self.retrieval_model.retrieve()
        positives = self.labels["experts"].values
        assert len(positives) == len(user_predictions)
        filtered_hard_negative = [list(set(user_predictions[i]).difference(positives[i])) for i in
                                  range(len(positives))]
        return [self.user_string.loc[negative_id, 0].values.tolist() for negative_id in filtered_hard_negative]

    def train(self):
        if self.hard_neg:
            hard_negatives = self.retrieve_hard_negative()
        else:
            hard_negatives = None
        user_text = self.labels["experts"].map(lambda x: self.user_string.loc[x, 0].values.tolist()).to_list()
        pubs_text = self.pubs_string.loc[self.labels["pub_id"].values, 0].to_list()
        self.model.train(pubs_text, user_text, hard_negatives)

        return self

    def evaluate(self):
        user_text = self.labels["experts"].map(lambda x: self.user_string.loc[x, 0].values.tolist()).to_list()
        pubs_text = self.pubs_string.loc[self.labels["pub_id"].values, 0].to_list()

        self.model.evaluate(pubs_text, user_text, self.user_string.iloc[:, 0].values.tolist())

        return self


if __name__ == '__main__':
    BiEncoderRetrievalTrain("paraphrase-TinyBERT-L6-v2", hard_neg=False, device="cuda:1").train()
