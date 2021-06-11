from ..data.data_transform import base_data_transformation, paper_data_transformation, get_all_pub_info, \
    paper_embedding_transformation
from ..data.dataset import Dataset
from ..model.retrieval import BM25, TFIDFRetrieval, FastEmbeddingRetrievalModel
from ..evaluation import accuracy_custom, mean_average_precision
from ..model.embedding import PersistEmbeddingModel


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
        self.base_data = Dataset().read_base_dataset()

        self.pubs, self.labels = Dataset().read_train_dataset()

        self.retrieval_method = retrieval_method

        self.transformation_kwargs = transformation_kwargs if transformation_kwargs is not None else {}
        self.retrieval_kwargs = retrieval_kwargs if retrieval_kwargs is not None else {}

        self.setup()

    def setup(self):
        data_source = (base_data_transformation(self.base_data, **self.transformation_kwargs), self.base_data["id"])

        if self.retrieval_method == "bm25":
            self.retrieve_model = BM25
        elif self.retrieval_method == "tfidf":
            self.retrieve_model = TFIDFRetrieval
        self.retrieve_model = self.retrieve_model(data_source, **self.retrieval_kwargs)

    def retrieve(self):
        query = paper_data_transformation(self.pubs, **self.transformation_kwargs)
        return self.retrieve_model.retrieve_data(query)

    def evaluate(self):
        prediction = self.retrieve()
        return {"map": mean_average_precision(prediction, self.labels["experts"].values),
                "acc_recall": accuracy_custom(prediction, self.labels["experts"].values)}

    def close(self):
        self.retrieve_model.reset_database()


class EmbeddingRetrieve:
    def __init__(self,
                 transformation_kwargs: dict = None,
                 retrieval_kwargs: dict = None):
        """

        :param dataset_name:
        :param transformation:
        :param retrieval_method: [bm25, tfidf, embedding]
        :param transformation_kwargs:
        :param retrieval_kwargs:
        """
        self.base_data = Dataset().read_base_dataset()

        self.pubs, self.labels = Dataset().read_train_dataset()

        self.transformation_kwargs = transformation_kwargs if transformation_kwargs is not None else {}
        self.retrieval_kwargs = retrieval_kwargs if retrieval_kwargs is not None else {}

        self.cache_names = {
            "keywords": "keywords.pk",
            "title": "title.pk",
            "abstract": "abstract.pk",
            "prfs": "prfs.pk"
        }

        self.setup()

    def setup(self):
        prfs = base_data_transformation(self.base_data, **self.transformation_kwargs).values.tolist()

        keywords, title, abstract = get_all_pub_info(self.pubs, **self.transformation_kwargs)

        k_emb = PersistEmbeddingModel(self.cache_names["keywords"],
                                      {"model_name": "sentence-transformers/paraphrase-TinyBERT-L6-v2"}).load(keywords)
        t_emb = PersistEmbeddingModel(self.cache_names["title"],
                                      {"model_name": "sentence-transformers/paraphrase-TinyBERT-L6-v2"}).load(title)
        a_emb = PersistEmbeddingModel(self.cache_names["abstract"],
                                      {"model_name": "sentence-transformers/paraphrase-TinyBERT-L6-v2"}).load(abstract)

        p_emb = PersistEmbeddingModel(self.cache_names["prfs"],
                                      {"model_name": "sentence-transformers/paraphrase-TinyBERT-L6-v2"}).load(prfs)

        self.pubs_embedding = paper_embedding_transformation(k_emb, t_emb, a_emb, **self.transformation_kwargs)
        self.retrieve_model = FastEmbeddingRetrievalModel((p_emb, self.base_data["id"]), **self.retrieval_kwargs)

    def retrieve(self):
        query = self.pubs_embedding
        return self.retrieve_model.retrieve_data(query)

    def evaluate(self):
        prediction = self.retrieve()
        return {"map": mean_average_precision(prediction, self.labels["experts"].values),
                "acc_recall": accuracy_custom(prediction, self.labels["experts"].values)}

    def close(self):
        self.retrieve_model.reset_database()


if __name__ == '__main__':
    r = Retrieve("", "", "bm25")
    # print(r.evaluate())
    r.close()
