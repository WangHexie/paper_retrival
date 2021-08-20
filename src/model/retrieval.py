import itertools
import os
import pickle
from pathlib import Path
from typing import List, Tuple
from elasticsearch import Elasticsearch
from tqdm import tqdm
from elasticsearch.helpers import parallel_bulk
import math
from elasticsearch_dsl import MultiSearch, Search
import pysparnn.cluster_index as ci
from sklearn import feature_extraction
import numpy as np
import hnswlib
from sentence_transformers import LoggingHandler, util, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ..config import root_path


class BaseRetrieval:
    def __init__(self, data_source):
        self.data_source = data_source
        self.document, self.index = self.data_source

    def retrieve_data(self, query):
        pass


class SentenceSearch:
    def __init__(self, data: List[str], index=None):
        """
        :param data: [(str, index)]
        """
        self.data = data
        self.index = list(range(len(self.data))) if index is None else index

    def _build_index(self):
        pass

    def search_data(self, queries, k=50):
        """
        :param queries:
        :param k:
        :return: index of document
        """
        return []


class TfidfAnnSearch(SentenceSearch):
    def __init__(self, data: List[str], index=None, max_features=5000):
        super().__init__(data, index)
        self.max_features = max_features
        self._build_index()

    def _build_index(self):
        print("building search engine index")

        print("start train tfidf")
        #         self.tv = TfidfVectorizer(max_df=0.7, min_df=10)
        self.tv = feature_extraction.text.TfidfVectorizer(max_features=self.max_features)
        self.tv.fit(self.data)

        print("start transform tfidf")
        features_vec = self.tv.transform(self.data)

        # build the search index!
        print("start build index")
        self.cp = ci.MultiClusterIndex(features_vec, self.index)
        print("build finished")

    def search_data(self, queries: List[str], k=50):
        search_features_vec = self.tv.transform(queries)

        return self.cp.search(search_features_vec, k=k, k_clusters=2, return_distance=False)


class BM25(BaseRetrieval):
    def __init__(self, data_source: Tuple[List[str or dict], List[str]], index_name="pub", k=100, save=True,
                 retrieve_by_score=False, temp_k=150, normalize_score=False, normalize_method="max",
                 output_weight=False, length_normalize_func="linear", **kwargs):
        """
        test
        :param data_source:
        :param index_name:
        :param k:
        :param save:
        :param retrieve_by_score:
        :param temp_k:
        :param normalize_score:
        :param normalize_method: "max", "query_length"
        :param output_weight:
        :param length_normalize_func: "linear", "log"
        :param kwargs:
        """
        super().__init__(data_source)
        self.length_normalize_func = (lambda x: x) if length_normalize_func == "linear" else (
            lambda x: math.log((1 + x), 10))
        self.output_weight = output_weight
        self.normalize_method = normalize_method
        self.normalize_score = normalize_score
        self.temp_k = temp_k
        self.retrieve_by_score = retrieve_by_score
        self.es = Elasticsearch(timeout=18000)
        self.index_name = index_name
        self.k = k
        if save:
            self.save_to_database()

    def retrieve_data(self, query: List[str], boost_scores=None):

        if self.retrieve_by_score:
            result = self.multi_search(self.es, self.index_name, query, self.temp_k,
                                       add_score=True,
                                       field_boost_scores=boost_scores)  # temp_k :make sure we have high recall, we can higher this later
            quantile = 1 - (self.k / self.temp_k)

            scores = list(map(lambda x: [i[-1] for i in x], result))
            ids = list(map(lambda x: [i[0] for i in x], result))

            # do not normalize score or ....
            if self.normalize_method == "max":
                max_score = list(map(lambda x: max(x), scores))
            elif self.normalize_method == "query_length":
                max_score = list(map(lambda x: len(x.split()), query))
            elif not self.normalize_score:
                max_score = [1] * len(query)
            else:
                raise Exception("normalize_method not defined")

            max_score = list(map(lambda x: self.length_normalize_func(x), max_score))

            normalized_result = list(map(lambda x: [i / x[1] for i in x[0]], zip(scores, max_score)))
            to_full_score = list(itertools.chain.from_iterable(normalized_result))
            threshold = np.quantile(to_full_score, quantile)

            if self.output_weight:
                result = [list(filter(lambda x: x[1] > threshold, zip(id_list, score_list))) for
                          id_list, score_list in zip(ids, normalized_result)]
            else:
                result = [list(map(lambda x: x[0], filter(lambda x: x[1] > threshold, zip(id_list, score_list)))) for
                          id_list, score_list in zip(ids, normalized_result)]

        else:
            result = self.multi_search(self.es, self.index_name, query, self.k, field_boost_scores=boost_scores)

        return result

    @staticmethod
    def multi_search(es, index_name, query_texts: list, k=100, chunk=150, add_score=False,
                     field_boost_scores: dict = None):
        results = []

        for epoch in tqdm(range(math.ceil(len(query_texts) / chunk))):
            ms = MultiSearch(index=index_name).using(es)
            for i in range(epoch * chunk,
                           (epoch + 1) * chunk if (epoch + 1) * chunk < len(query_texts) else len(query_texts)):

                if field_boost_scores is None:
                    body = {"query":
                                {"match": {"content": query_texts[i]}},
                            "size": k
                            }
                else:
                    body = {"query": {"multi_match":
                        {
                            "query": query_texts[i],
                            "fields": [key + "^" + str(value) for key, value in field_boost_scores.items()]
                        }
                    },
                        "size": k
                    }

                s = Search().update_from_dict(body)
                ms = ms.add(s)
            responses = ms.execute()
            for response in responses:
                if add_score:
                    results.append([[hit.meta["id"], hit.meta["score"]] for hit in response.hits])
                else:
                    results.append([hit.meta["id"] for hit in response.hits])
            del responses
        return results

    @staticmethod
    def parallel_set_up_database(es, documents, indexes, index_name):
        assert len(documents) == len(indexes)

        def gendata(documents_str, indexes_str, index_name_str):
            for i in tqdm(range(len(documents_str))):

                if type(documents_str[i]) == str:

                    doc = {
                        'content': documents_str[i]
                    }
                else:
                    doc = documents_str[i]

                yield {
                    '_op_type': 'index',
                    '_index': index_name_str,
                    '_type': 'document',
                    '_id': indexes_str[i],
                    '_source': doc
                }

        result = []
        for success, info in parallel_bulk(es, gendata(documents, indexes, index_name)):
            if not success:
                result.append(info)
        return result

    def save_to_database(self):
        self.parallel_set_up_database(self.es, self.document, self.index, self.index_name)
        return self

    def reset_database(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        return self


class TFIDFRetrieval(BaseRetrieval):
    def __init__(self, data_source: Tuple[List[str], List[str]], k=100, number_of_features=5000, **kwargs):
        super().__init__(data_source)
        self.kwargs = kwargs
        self.k = k
        self.retrieval_model = TfidfAnnSearch(self.document, self.index, max_features=number_of_features)

    def retrieve_data(self, query):
        return self.retrieval_model.search_data(query, k=self.k)

    def reset_database(self):
        return self


class FastEmbeddingRetrievalModel(BaseRetrieval):
    def __init__(self, data_source: Tuple[np.array, List[str]], k=100, **kwargs):
        super().__init__(data_source)
        self.k = k
        self.embedding = self.document

        self.index_path = str(Path(root_path, "cache", "index.bin"))

        self._index_init()

        # self._load_index()

    # def _save_embedding(self):
    #     embeddings = self.batch_convert.batch_convert(self.documents.get_paragraph())
    #
    #     with open(self.embedding_path, "wb") as f:
    #         pickle.dump(embeddings, f)
    #
    # def load_embedding(self):
    #     with open(self.embedding_path, "rb") as f:
    #         self.embedding = pickle.load(f)

    def _index_init(self):
        dim = len(self.embedding[0])
        num_elements = 5000000

        batch_size = 1000

        self.p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

        self.p.init_index(max_elements=num_elements, ef_construction=200, M=32)
        indexes = list(range(len(self.index)))

        length = math.ceil(len(self.embedding) / batch_size)
        for i in tqdm(range(length)):
            self.p.add_items(self.embedding[i * batch_size:(i + 1) * batch_size],
                             indexes[i * batch_size:(i + 1) * batch_size])
            # del self.embedding[:batch_size]

        # Controlling the recall by setting ef:
        self.p.set_ef(50)  # ef should always be > k

        del self.embedding
        self.p.save_index(self.index_path)

    # def _load_index(self):
    #     self.p = hnswlib.Index(space='cosine',
    #                            dim=768)  # the space can be changed - keeps the data, alters the distance function.
    #
    #     print("\nLoading index from 'first_half.bin'\n")
    #
    #     # Increase the total capacity (max_elements), so that it will handle the new data
    #     self.p.load_index(self.index_path)

    def retrieve_data(self, query: np.array):
        labels, distances = self.p.knn_query(query, k=self.k)

        return [self.index[i] for i in labels]

    def reset_database(self):
        return self


class BiEncoder:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


class BertRankSE:
    def __init__(self, batch_size=8, num_epochs=1, model_save_path=os.path.join(root_path, "models"), max_length=512,
                 initial_load=True):
        self.max_length = max_length
        self.model_save_path = model_save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        model_name = 'allenai/scibert_scivocab_uncased'
        if initial_load:
            try:
                self.model = SentenceTransformer(os.path.join(self.model_save_path, model_name), device="cuda:1")
            except:
                self.model = SentenceTransformer(model_name, device="cuda:1",
                                                 cache_folder=os.path.join(root_path, "models"))
            self.model.save(os.path.join(self.model_save_path, model_name))

    def load_model(self):
        self.model = SentenceTransformer(os.path.join(root_path, "models", "manual_save"), device="cuda:1")

        return self

    @staticmethod
    def _reformat_example(paper_text, user_text, labels):
        assert len(paper_text) == len(user_text) and len(paper_text) == len(labels)
        # breakpoint()
        train_examples = [InputExample(texts=[paper[0], user[0]], label=y) for paper, user, y in
                          zip(paper_text, user_text, labels)]
        return train_examples

    def train(self, paper_text, user_text, y):
        data = self._reformat_example(paper_text, user_text, y)
        train_data, test_data = train_test_split(data, shuffle=True, random_state=8888)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        # test_dataloader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_data, name='Quora-dev')
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up

        model_save_path = os.path.join(self.model_save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.model.fit(train_dataloader=train_dataloader,
                       evaluator=evaluator,
                       epochs=self.num_epochs,
                       evaluation_steps=5000,
                       warmup_steps=warmup_steps,
                       output_path=model_save_path)
        self.model.save(os.path.join(self.model_save_path, "manual_save"))
        # self.model.save_model("../../../models")
        return self

    def predict(self, paper_text, user_text):
        return self.model.predict(list(zip(paper_text, user_text)), show_progress_bar=True)

    @staticmethod
    def convert_prediction_to_dictionary(paper_id, user_id, prediction, threshold=0.3):
        assert len(paper_id) == len(user_id) and len(paper_id) == len(prediction)
        result = dict([(i, []) for i in set(paper_id)])
        for paper, user, score in zip(paper_id, user_id, prediction):
            if score > threshold:
                result[paper].append(user)

        return result


if __name__ == '__main__':
    print(BM25((["dafasdfa"], ["fdasfa "])).reset_database())
    # print(TFIDFRetrieval((["dafasdfa"], ["fdasfa "])).retrieve_data(["da"]))
