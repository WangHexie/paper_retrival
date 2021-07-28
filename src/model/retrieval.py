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
    def __init__(self, data_source: Tuple[List[str], List[str]], index_name="pub", k=100, save=True, retrieve_by_score=False, temp_k=150, **kwargs):
        super().__init__(data_source)
        self.temp_k = temp_k
        self.retrieve_by_score = retrieve_by_score
        self.es = Elasticsearch(timeout=18000)
        self.index_name = index_name
        self.k = k
        if save:
            self.save_to_database()

    def retrieve_data(self, query: List[str]):

        if self.retrieve_by_score:
            result = self.multi_search(self.es, self.index_name, query, self.temp_k, add_score=True) # temp_k :make sure we have high recall, we can higher this later
            quantile = 1 - (self.k / self.temp_k)
            to_full_score = list(map(lambda x: x[-1], itertools.chain.from_iterable(result)))
            threshold = np.quantile(to_full_score, quantile)

            result = [list(map(lambda x:x[0], list(filter(lambda x:x[-1] > threshold, recalled_result )))) for recalled_result in result]

        else:
            result = self.multi_search(self.es, self.index_name, query, self.k)

        return result

    @staticmethod
    def multi_search(es, index_name, query_texts: list, k=100, chunk=150, add_score=False):
        results = []

        for epoch in tqdm(range(math.ceil(len(query_texts) / chunk))):
            ms = MultiSearch(index=index_name).using(es)
            for i in range(epoch * chunk,
                           (epoch + 1) * chunk if (epoch + 1) * chunk < len(query_texts) else len(query_texts)):
                body = {"query":
                            {"match": {"content": query_texts[i]}},
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
                doc = {
                    'content': documents_str[i]
                }

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
        # self.es.indices.put_settings({
        #     self.index_name: {
        #         "max_clause_count": 2
        #     }
        # })
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
            self.p.add_items(self.embedding[i * batch_size:(i + 1) * batch_size], indexes[i * batch_size:(i + 1) * batch_size])
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


if __name__ == '__main__':
    print(BM25((["dafasdfa"], ["fdasfa "])).reset_database())
    # print(TFIDFRetrieval((["dafasdfa"], ["fdasfa "])).retrieve_data(["da"]))
