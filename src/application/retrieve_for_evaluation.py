from ..data.data_transform import base_data_transformation, paper_data_transformation, get_all_pub_info, \
    paper_embedding_transformation, base_data_transformation_to_diction_records
from ..data.dataset import Dataset
from ..model.retrieval import BM25, TFIDFRetrieval, FastEmbeddingRetrievalModel, PersistSentenceBertModel
from ..evaluation import accuracy_custom, mean_average_precision
from ..model.embedding import PersistEmbeddingModel
from ..model.graph import Graph


class Retrieve:
    def __init__(self,
                 dataset_name: str,
                 transformation: str,
                 retrieval_method: str,
                 refine_retrieved_result: bool or str = False,
                 transformation_kwargs: dict = None,
                 retrieval_kwargs: dict = None,
                 refine_kwargs: dict = None,
                 hyper_param_search=False,
                 evaluation_num=None,
                 prediction=False
                 ):
        """

        :param dataset_name:
        :param transformation:
        :param refine_retrieved_result: [False, "link"]
        :param retrieval_method: [bm25, tfidf, embedding]
        :param transformation_kwargs:
        :param retrieval_kwargs:
        :param refine_kwargs:
        """
        self.evaluation_num = evaluation_num
        self.hyper_param_search = hyper_param_search
        self.refine_kwargs = refine_kwargs
        self.refine_retrieved_result = refine_retrieved_result
        self.base_data = Dataset().read_base_dataset()

        if prediction:
            self.pubs = Dataset().read_valid_dataset()
        else:
            self.pubs, self.labels = Dataset().read_train_dataset()

        self.retrieval_method = retrieval_method

        self.transformation_kwargs = transformation_kwargs if transformation_kwargs is not None else {}
        self.retrieval_kwargs = retrieval_kwargs if retrieval_kwargs is not None else {}

        self._setup()

    def _setup(self):
        if not self.hyper_param_search:
            data_source = (base_data_transformation(self.base_data, **self.transformation_kwargs), self.base_data["id"])
        else:
            data_source = (base_data_transformation_to_diction_records(self.base_data, **self.transformation_kwargs),
                           self.base_data["id"])

        if self.retrieval_method == "bm25":
            self.retrieve_model = BM25
        elif self.retrieval_method == "tfidf":
            self.retrieve_model = TFIDFRetrieval
        self.retrieve_model = self.retrieve_model(data_source, **self.retrieval_kwargs)

        if self.refine_retrieved_result == "link":
            self.refinement_method = Graph(**self.refine_kwargs)
            self.refinement_method.fit(self.base_data)

        self.query = paper_data_transformation(self.pubs, **self.transformation_kwargs)

    def retrieve(self, boost_scores=None):
        if self.refine_retrieved_result is not False:
            return self.refinement_method.retrieve(
                self.retrieve_model.retrieve_data(self.query[:self.evaluation_num], boost_scores))
        else:
            return self.retrieve_model.retrieve_data(self.query[:self.evaluation_num], boost_scores)

    def evaluate_by_prediction(self, prediction):
        return {"map": mean_average_precision(prediction, self.labels["experts"].values[:len(prediction)]),
                "acc_recall": accuracy_custom(prediction, self.labels["experts"].values[:len(prediction)]),
                "length": sum([len(i) for i in prediction]) / len(prediction)}

    def evaluate(self, boost_scores=None):
        prediction = self.retrieve(boost_scores)
        return self.evaluate_by_prediction(prediction)

    def close(self):
        self.retrieve_model.reset_database()


class EmbeddingRetrieve:
    def __init__(self,
                 transformation_kwargs: dict = None,
                 retrieval_kwargs: dict = None,
                 model_kwargs: dict = None,
                 model_type="sentencebert"):
        """

        :param transformation_kwargs:
        :param retrieval_kwargs:
        """
        self.model_kwargs = model_kwargs
        self.model = PersistSentenceBertModel if model_type == "sentencebert" else PersistEmbeddingModel
        self.model_type = model_type
        self.base_data = Dataset().read_base_dataset()

        self.pubs, self.labels = Dataset().read_train_dataset()

        self.transformation_kwargs = transformation_kwargs if transformation_kwargs is not None else {}
        self.retrieval_kwargs = retrieval_kwargs if retrieval_kwargs is not None else {}

        self.cache_names = {
            "keywords": "keywords.pk",
            "title": "title.pk",
            "abstract": "abstract.pk",
            "prfs": "prfs.pk",
            "pubs": "pubs.pk"
        }

        self.setup()

    def setup(self):
        prfs = base_data_transformation(self.base_data, **self.transformation_kwargs).values.tolist()

        if self.model_type == "sentencebert":
            pubs_string = paper_data_transformation(self.pubs, add_abstract=True, add_title=True, abstract_length=200)
            user_string = base_data_transformation(self.base_data, log_transform=True)
            print("encoding pubs")
            self.pubs_embedding = self.model(self.model_kwargs["model_name"]+self.cache_names["pubs"], self.model_kwargs).load(pubs_string.values)
            print("encoding users")
            self.p_emb = self.model(self.model_kwargs["model_name"]+self.cache_names["prfs"], self.model_kwargs).load(user_string.values)
        else:

            keywords, title, abstract = get_all_pub_info(self.pubs, **self.transformation_kwargs)

            k_emb = self.model(self.cache_names["keywords"], self.model_kwargs).load(keywords)
            t_emb = self.model(self.cache_names["title"], self.model_kwargs).load(title)
            a_emb = self.model(self.cache_names["abstract"], self.model_kwargs).load(abstract)

            p_emb = self.model(
                self.cache_names["prfs"] + "logs_transform_" + str(self.transformation_kwargs["log_transform"]),
                self.model_kwargs).load(prfs)

            self.pubs_embedding = paper_embedding_transformation(k_emb, t_emb, a_emb, **self.transformation_kwargs)
            self.p_emb = p_emb
        self.retrieve_model = FastEmbeddingRetrievalModel((self.p_emb, self.base_data["id"]), **self.retrieval_kwargs)

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
