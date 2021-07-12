import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

from .retrieve_for_evaluation import Retrieve
from ..data.data_transform import paper_data_transformation, base_data_transformation
from ..model.rerank import BertRerank, BertRankSE


class BertRerankApplication:
    def __init__(self):
        self.retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=60), transformation_kwargs=dict(add_title=True))

        self.rerank_model = BertRankSE()

        self.pubs = self.retrieval_model.pubs
        self.user_info = self.retrieval_model.base_data
        self.labels = self.retrieval_model.labels

        self._setup()

    def _setup(self):
        pubs_string = paper_data_transformation(self.pubs, add_abstract=True, add_title=True, abstract_length=200)
        user_string = base_data_transformation(self.user_info, log_transform=True)

        # warning: data reshape????
        self.user_string = pd.DataFrame(data=user_string.values, index=self.user_info["id"])
        self.pubs_string = pd.DataFrame(data=pubs_string.values, index=self.pubs["id"])

        return self

    # def _train_model(self):

    def _reformat_data(self, pubs_string, user_ids, labels):
        labels = labels["experts"].values
        pubs_string = pubs_string.values

        assert len(pubs_string) == len(user_ids) and len(pubs_string) == len(labels)
        print(len(pubs_string), len(user_ids), len(labels))
        # breakpoint()
        full_pubs_string = list(itertools.chain.from_iterable([[pub] * len(user_id) for pub, user_id in zip(pubs_string, user_ids)]))
        full_user_string = list(itertools.chain.from_iterable(
            [self.user_string.loc[ids].values.tolist() for ids in user_ids]))

        y = [1 if user_id in label_list else 0 for ids, label_list in zip(user_ids, labels) for user_id in ids]


        print(len(full_user_string), len(full_pubs_string), len(y))

        assert len(full_user_string) == len(full_pubs_string) and len(full_user_string) == len(y)

        return full_pubs_string, full_user_string, y

    def rerank(self):
        user_predictions = self.retrieval_model.retrieve()
        retrieve_metrics = self.retrieval_model.evaluate_by_prediction(user_predictions)
        print(retrieve_metrics)

        train_pubs_string, test_pubs_string, train_user_prediction, test_user_prediction, train_labels, test_labels = train_test_split(
            self.pubs_string, user_predictions, self.labels, shuffle=True, random_state=8888)
        del self.retrieval_model
        del self.pubs
        del self.user_info
        del self.pubs_string
        # del self.user_string
        del user_predictions

        self.rerank_model.train(*self._reformat_data(train_pubs_string, train_user_prediction, train_labels))

        # print(self.rerank_model.evaluate(*self._reformat_data(test_pubs_string, test_user_prediction, test_labels)))


if __name__ == '__main__':
    BertRerankApplication().rerank()
