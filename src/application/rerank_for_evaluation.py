import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools

from .retrieve_for_evaluation import Retrieve
from ..data.data_transform import paper_data_transformation, base_data_transformation
from ..model.rerank import BertRerank, BertRankSE
import json


class BertRerankApplication:
    def __init__(self, model_name, k=60, prediction=False):
        self.retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=k),
                                        transformation_kwargs=dict(add_abstract=False,
                                                                   add_title=True,
                                                                   add_paper_keywords=True, add_prsf_interest=True,
                                                                   add_paper_title=True, abstract_length=200),
                                        prediction=prediction,
                                        hyper_param_search=True)

        self.rerank_model = BertRankSE(model_name, initial_load=True)
        self.prediction = prediction
        self.pubs = self.retrieval_model.pubs
        self.user_info = self.retrieval_model.base_data
        if not self.prediction:
            self.labels = self.retrieval_model.labels

        self._setup()

    def _setup(self):
        pubs_string = paper_data_transformation(self.pubs, add_abstract=True, add_title=True, abstract_length=200)
        user_string = base_data_transformation(self.user_info, log_transform=True, add_abstract=True, add_title=True,
                                               add_paper_keywords=True, add_prsf_interest=True, add_paper_title=True)

        # warning: data reshape????
        self.user_string = pd.DataFrame(data=user_string.values, index=self.user_info["id"])
        self.pubs_string = pd.DataFrame(data=pubs_string.values, index=self.pubs["id"])

        return self

    # def _train_model(self):

    def _reformat_data(self, pubs_string, user_ids, true_user_ids=None):

        pubs_string = pubs_string.values

        full_pubs_string = list(
            itertools.chain.from_iterable([[pub] * len(user_id) for pub, user_id in zip(pubs_string, user_ids)]))
        full_user_string = list(itertools.chain.from_iterable(
            [self.user_string.loc[ids].values.tolist() for ids in user_ids]))

        if true_user_ids is not None:
            true_user_ids = true_user_ids["experts"].values
            user_ids = [list(set(list(retrieved_data) + list(true_label))) for retrieved_data, true_label in
                        zip(user_ids, true_user_ids)]
            y = [1 if user_id in label_list else 0 for ids, label_list in zip(user_ids, true_user_ids) for user_id in
                 ids]
            return [i[0] for i in full_pubs_string], [i[0] for i in full_user_string], y
        else:
            full_pubs_id = list(itertools.chain.from_iterable(
                [[pub] * len(user_id) for pub, user_id in zip(self.pubs["id"].values, user_ids)]))
            return [i[0] for i in full_pubs_string], [i[0] for i in full_user_string], full_pubs_id

    def train(self):
        user_predictions = self.retrieval_model.retrieve(boost_scores=dict(
            prfs_interests=0.8,
            prfs_keywords=0.8,
            pub_title=1,
            pub_keywords=0.87,
        ))
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

    def predict(self, test_length=None):
        if test_length is None:
            test_length = [10, 23, 30]
        user_predictions = self.retrieval_model.retrieve(boost_scores=dict(
            prfs_interests=0.8,
            prfs_keywords=0.8,
            pub_title=1,
            pub_keywords=0.87,
        ))
        full_pubs_string, full_user_string, full_pubs_id = self._reformat_data(self.pubs_string, user_predictions)
        prediction = self.rerank_model.predict(full_pubs_string, full_user_string)

        with open("rerank_prediction.json", "wb") as f:
            pickle.dump(prediction, f)

        with open("rerank_user_prediction.json", "wb") as f:
            pickle.dump(user_predictions, f)

        length = len(user_predictions)
        # test_length = [10,23,30]
        full_test_length = [i * length for i in test_length]
        quantiles = [1 - i / len(full_pubs_string) for i in full_test_length]
        thresholds = [np.quantile(prediction, quantile) for quantile in quantiles]
        print(list(zip(test_length, thresholds)))

        return [self.rerank_model.convert_prediction_to_dictionary(full_pubs_id,
                                                                   list(
                                                                       itertools.chain.from_iterable(user_predictions)),
                                                                   prediction, threshold=threshold) for threshold in
                thresholds]

        # print(self.rerank_model.evaluate(*self._reformat_data(test_pubs_string, test_user_prediction, test_labels)))


# class BertRerankPrediction:
#     def __init__(self, model_name, k=60):
#         self.retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=k),
#                                         transformation_kwargs=dict(add_abstract=False,
#                                                                    add_title=True,
#                                                                    add_paper_keywords=True, add_prsf_interest=True,
#                                                                    add_paper_title=True, abstract_length=200),
#                                         prediction=True,
#                                         hyper_param_search=True)
#
#         self.rerank_model = BertRankSE(model_name, initial_load=True)
#
#         self.pubs = self.retrieval_model.pubs
#         self.user_info = self.retrieval_model.base_data
#
#         self._setup()
#
#     def _setup(self):
#         pubs_string = paper_data_transformation(self.pubs, add_abstract=True, add_title=True, abstract_length=200)
#         user_string = base_data_transformation(self.user_info, log_transform=True, add_abstract=True, add_title=True,
#                                                add_paper_keywords=True, add_prsf_interest=True, add_paper_title=True)
#
#         # warning: data reshape????
#         self.user_string = pd.DataFrame(data=user_string.values, index=self.user_info["id"])
#         self.pubs_string = pd.DataFrame(data=pubs_string.values, index=self.pubs["id"])
#
#         return self
#
#     # def _train_model(self):
#
#     def _reformat_data(self, pubs_string, user_ids):
#         pubs_string = pubs_string.values
#
#         assert len(pubs_string) == len(user_ids)
#         print(len(pubs_string), len(user_ids))
#         # breakpoint()
#         full_pubs_string = list(
#             itertools.chain.from_iterable([[pub[0]] * len(user_id) for pub, user_id in zip(pubs_string, user_ids)]))
#         full_pubs_id = list(itertools.chain.from_iterable(
#             [[pub] * len(user_id) for pub, user_id in zip(self.pubs["id"].values, user_ids)]))
#         full_user_string = list(itertools.chain.from_iterable(
#             [self.user_string.loc[ids].values.tolist() for ids in user_ids]))
#
#         print(len(full_user_string), len(full_pubs_string))
#
#         assert len(full_user_string) == len(full_pubs_string)
#
#         return full_pubs_string, [i[0] for i in full_user_string], full_pubs_id
#
#     def rerank(self, test_length=None):
#         if test_length is None:
#             test_length = [10, 23, 30]
#         user_predictions = self.retrieval_model.retrieve(boost_scores=dict(
#             prfs_interests=0.8,
#             prfs_keywords=0.8,
#             pub_title=1,
#             pub_keywords=0.87,
#         ))
#         full_pubs_string, full_user_string, full_pubs_id = self._reformat_data(self.pubs_string, user_predictions)
#         prediction = self.rerank_model.predict(full_pubs_string, full_user_string)
#
#         with open("rerank_prediction.json", "w") as f:
#             json.dump(prediction, f)
#
#         with open("rerank_user_prediction.json", "w") as f:
#             json.dump(user_predictions, f)
#
#         length = len(user_predictions)
#         # test_length = [10,23,30]
#         full_test_length = [i * length for i in test_length]
#         quantiles = [1 - i / len(full_pubs_string) for i in full_test_length]
#         thresholds = [np.quantile(prediction, quantile) for quantile in quantiles]
#         print(list(zip(test_length, thresholds)))
#
#         return [self.rerank_model.convert_prediction_to_dictionary(full_pubs_id,
#                                                                    list(
#                                                                        itertools.chain.from_iterable(user_predictions)),
#                                                                    prediction, threshold=threshold) for threshold in
#                 thresholds]



if __name__ == '__main__':
    BertRerankApplication("scibert_24h", k=60, prediction=True).predict()
