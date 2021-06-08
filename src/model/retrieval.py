from typing import List, Tuple
from elasticsearch import Elasticsearch
from tqdm import tqdm
from elasticsearch.helpers import parallel_bulk
import math
from elasticsearch_dsl import MultiSearch, Search
import pysparnn.cluster_index as ci
from sklearn import feature_extraction


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
    def __init__(self, data_source: Tuple[List[str], List[str]], index_name="pub", k=100, save=True, **kwargs):
        super().__init__(data_source)
        self.es = Elasticsearch(timeout=18000)
        self.index_name = index_name
        self.k = k
        if save:
            self.save_to_database()

    def retrieve_data(self, query: List[str]):
        result = self.multi_search(self.es, self.index_name, query, self.k)
        return result

    @staticmethod
    def multi_search(es, index_name, query_texts: list, k=100, chunk=100):
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


if __name__ == '__main__':
    # print(BM25((["dafasdfa"], ["fdasfa "])).reset_database().save_to_database().retrieve_data(["da"]))
    print(TFIDFRetrieval((["dafasdfa"], ["fdasfa "])).retrieve_data(["da"]))
