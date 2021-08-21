import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

from src.application.retrieve_for_evaluation import Retrieve
from src.data.data_transform import paper_data_transformation, base_data_transformation
from src.model.retrieval import BiEncoderRetrieval


class BiEncoderRetrievalTrain:
    def __init__(self, model_name, device="cuda:0"):
        retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=60, save=False),
                                   transformation_kwargs=dict(add_title=True))

        self.model = BiEncoderRetrieval(model_name=model_name, batch_size=32, device=device)

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

    def evaluate(self):
        user_text = self.labels["experts"].map(lambda x: [i for i in itertools.chain.from_iterable(
            self.user_string.loc[self.labels["experts"][0]].values.tolist())]).to_list()
        pubs_text = self.pubs_string.loc[self.labels["pub_id"].values, 0].to_list()

        self.model.evaluate(pubs_text, user_text, self.pubs_string.iloc[:, 0].values.to_list())

        return self




if __name__ == '__main__':
    BiEncoderRetrievalTrain("triplet").evaluate()
