from ..data.data_transform import base_data_transformation, paper_data_transformation
from ..data.dataset import Dataset
from ..model.retrieval import BM25, TFIDFRetrieval


class Retrieve:
    def __init__(self,
                 dataset_name: str,
                 transformation: str,
                 retrieval_method: str,
                 transformation_kwargs: dict = None,
                 retrieval_kwargs: dict = None):
        """

        :param dataset_name:
        :param transformation:
        :param retrieval_method: [bm25, tfidf, embedding]
        :param transformation_kwargs:
        :param retrieval_kwargs:
        """
        base_data = Dataset().read_base_dataset()

        pubs = Dataset().read_train_dataset()
        if retrieval_method == "bm25":
            retrieve_model  = BM25
        elif retrieval_method == "tfidf":
            retrieve_model = TFIDFRetrieval


