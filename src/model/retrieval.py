import itertools
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from elasticsearch import Elasticsearch
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm import tqdm
from elasticsearch.helpers import parallel_bulk
import math
from elasticsearch_dsl import MultiSearch, Search
import pysparnn.cluster_index as ci
from sklearn import feature_extraction
import numpy as np
import hnswlib
from sentence_transformers import LoggingHandler, util, SentenceTransformer, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.sampler.unique_sampler import NoduplicateSampler
from src.model.persist import PersistModel
from src.model.loss.infonce import InfoNCE
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


class BiEncoderRetrieval:
    def __init__(self, batch_size=8, num_epochs=1, model_save_path=os.path.join(root_path, "models"), max_length=512,
                 initial_load=True, loss="infoNce", num_of_neg=2, model_name="paraphrase-TinyBERT-L6-v2",
                 device="cuda:0",
                 multi_process=False):
        """

        :param batch_size:
        :param num_epochs:
        :param model_save_path:
        :param max_length:
        :param initial_load:
        :param loss: infonce or  MultipleNegativesRankingLoss
        :param num_of_neg:
        :param model_name:
        :param device:
        :param multi_process:
        """
        self.multi_process = multi_process
        self.device = device
        self.num_of_neg = num_of_neg
        self.loss = loss
        self.max_length = max_length
        self.model_save_path = model_save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_name = model_name
        # model_name = 'allenai/scibert_scivocab_uncased'
        if initial_load:
            try:
                self.model = SentenceTransformer(os.path.join(self.model_save_path, model_name), device=self.device)
            except:
                self.model = SentenceTransformer(model_name, device=self.device,
                                                 cache_folder=os.path.join(root_path, "models"))
                self.model.save(os.path.join(self.model_save_path, model_name))

    def load_model(self):
        self.model = SentenceTransformer(os.path.join(root_path, "models", "manual_save"), device=self.device)

        return self

    @staticmethod
    def _reformat_example_batch_triplet(paper_text: List[str], user_texts: List[List[str]]):
        assert len(paper_text) == len(user_texts)
        # breakpoint()

        paper_examples = [InputExample(texts=[paper_text[i]], label=i) for i in range(len(paper_text))]

        user_text_examples = list(itertools.chain.from_iterable(
            [[InputExample(texts=[text], label=i) for text in user_texts[i]] for i in range(len(user_texts))]))
        return paper_examples + user_text_examples

    def _reformat_example_infonce(self, paper_text: List[str], user_texts: List[List[str]]):
        assert len(paper_text) == len(user_texts)
        # breakpoint()
        full_user_text = list(itertools.chain.from_iterable(user_texts))

        def generate_negative_samples(pos_text, full_text, num_to_generate):
            neg_samples = random.sample(full_text, num_to_generate+15)
            return list(filter(lambda x: x not in pos_text, neg_samples))[:num_to_generate]

        data = [InputExample(texts=[paper_text[i],
                                    user_texts[i][user_index],
                                    *generate_negative_samples(user_texts[i], full_user_text, self.num_of_neg)],
                             label=i) for i in range(len(paper_text)) for user_index in range(len(user_texts[i]))]
        return data

    @staticmethod
    def _reformat_example_mnl(paper_text: List[str], user_texts: List[List[str]],
                              hard_negatives: List[List[str]]):
        assert len(paper_text) == len(user_texts) and len(user_texts) == len(hard_negatives)

        data = [InputExample(texts=[paper_text[i],
                                    user_texts[i][user_index],
                                    hard_negatives[i][user_index % len(hard_negatives[i])]],
                             label=i) for i in range(len(paper_text)) for user_index in range(len(user_texts[i]))]
        return data

    @staticmethod
    def construct_evaluator(paper_text: List[str], user_texts: List[List[str]], hard_negative: List[List[str]] = None):
        queries = {str(i): paper_text[i] for i in range(len(paper_text))}
        if hard_negative is not None:
            docs = {hash(doc): doc for doc in itertools.chain.from_iterable(user_texts + hard_negative)}
        else:
            docs = {hash(doc): doc for doc in itertools.chain.from_iterable(user_texts)}

        relevant_docs = {str(i): set([hash(text) for text in user_texts[i]]) for i in range(len(user_texts))}

        return InformationRetrievalEvaluator(queries=queries, corpus=docs, relevant_docs=relevant_docs,
                                             precision_recall_at_k=[1, 3, 5, 10, 23, 50, 100], show_progress_bar=True)

    def evaluate(self, paper_text, user_texts, full_texts):
        queries = {str(i): paper_text[i] for i in range(len(paper_text))}
        docs = {hash(doc): doc for doc in full_texts}
        relevant_docs = {str(i): set([hash(text) for text in user_texts[i]]) for i in range(len(user_texts))}
        evaluator = InformationRetrievalEvaluator(queries=queries, corpus=docs, relevant_docs=relevant_docs,
                                                  precision_recall_at_k=[1, 3, 5, 10, 23, 50, 100],
                                                  show_progress_bar=True)

        self.model.evaluate(evaluator, output_path=os.path.join(root_path, "evaluation_log", self.model_name))
        return self

    def train(self, paper_text, user_text, negative_text=None):
        if negative_text is not None:

            train_paper_txt, test_paper_text, train_user_text, text_user_text, train_negative, test_negative = train_test_split(
                paper_text, user_text, negative_text,
                shuffle=True,
                random_state=8888)
        else:
            train_paper_txt, test_paper_text, train_user_text, text_user_text = train_test_split(
                paper_text, user_text,
                shuffle=True,
                random_state=8888)
        if self.loss == "infoNce":
            train_data = self._reformat_example_infonce(train_paper_txt, train_user_text)
        elif self.loss == "MultipleNegativesRankingLoss":
            train_data = self._reformat_example_mnl(train_paper_txt, train_user_text, train_negative)
        else:
            train_data = self._reformat_example_batch_triplet(train_paper_txt, train_user_text)

        # breakpoint()

        if self.loss == "MultipleNegativesRankingLoss":
            sampler = NoduplicateSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=sampler, shuffle=False, drop_last=True,
                                          batch_size=self.batch_size)
        else:
            train_dataloader = DataLoader(train_data, batch_size=self.batch_size)

        if negative_text is not None:
            evaluator = self.construct_evaluator(test_paper_text, text_user_text, test_negative)
        else:
            evaluator = self.construct_evaluator(test_paper_text, text_user_text)
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up

        model_save_path = os.path.join(self.model_save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.loss == "infoNce":
            train_loss = InfoNCE(self.model)
        else:
            train_loss = losses.BatchHardSoftMarginTripletLoss(model=self.model)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=self.num_epochs,
                       evaluation_steps=5000,
                       warmup_steps=warmup_steps,
                       output_path=model_save_path)
        self.model.save(os.path.join(self.model_save_path, "manual_save"))
        # self.model.save_model("../../../models")
        return self

    def predict(self, texts):
        if self.multi_process:
            pools = self.model.start_multi_process_pool(["cuda:0", "cuda:1"])
            return self.model.encode_multi_process(texts, pools)
        else:
            return self.model.encode(texts, show_progress_bar=True)


class PersistSentenceBertModel:
    def __init__(self, cache_name, emd_model_kwargs):
        self.cache_name = cache_name
        self.emd_model_kwargs = emd_model_kwargs

    def load(self, document: List[str]):
        pm = PersistModel(self.cache_name)
        if pm.exist():
            return pm.load()
        else:
            emb = BiEncoderRetrieval(**self.emd_model_kwargs)
            embedding = emb.predict(document)
            pm.save(embedding)
            return embedding


if __name__ == '__main__':
    print(BM25((["dafasdfa"], ["fdasfa "])).reset_database())
    # print(TFIDFRetrieval((["dafasdfa"], ["fdasfa "])).retrieve_data(["da"]))
