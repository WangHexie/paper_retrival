import json
import os

from .rerank_for_evaluation import BertRerankApplication
from .retrieve_for_evaluation import Retrieve, EmbeddingRetrieve


def retrieval_output_prediction(k=5):
    retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=k),
                               transformation_kwargs=dict(add_abstract=False,
                                                          add_title=True,
                                                          add_paper_keywords=True, add_prsf_interest=True,
                                                          add_paper_title=True, abstract_length=200),
                               prediction=True,
                               hyper_param_search=True)
    data = retrieval_model.retrieve(boost_scores=dict(
        prfs_interests=0.8,
        prfs_keywords=0.8,
        pub_title=1,
        pub_keywords=0.87,
    ))
    pubs_id = retrieval_model.pubs["id"].values
    json_prediction = [{"pub_id": pub_id, "experts": experts} for pub_id, experts in zip(pubs_id, data)]
    if not os.path.exists("output"):
        os.mkdir("output")
    with open("output/prediction.json", "w") as f:
        json.dump(json_prediction, f)


def rerank_output_prediction():
    test_length = [10, 23, 30]
    results_list = BertRerankApplication("scibert_24h", k=60, prediction=True).predict(test_length=test_length)

    def save_prediction(results, k):
        json_prediction = [{"pub_id": pub_id, "experts": experts} for pub_id, experts in results.items()]
        if not os.path.exists("output"):
            os.mkdir("output")
        with open(f"output/prediction_rerank_{k}.json", "w") as f:
            json.dump(json_prediction, f)

    [save_prediction(result, k) for k, result in zip(test_length, results_list)]


def embedding_retrieval_output_prediction(k=10):
    from yaml_param_search import Config

    config = Config("./", "emb_retrieval_search_sentencebert_mnl_fix_prediction.yaml")
    for i in config.yield_param():
        retrieval_model = EmbeddingRetrieve(prediction=True, **i)
        data = retrieval_model.retrieve()
        pubs_id = retrieval_model.pubs["id"].values

        json_prediction = [{"pub_id": pub_id, "experts": experts.tolist()} for pub_id, experts in zip(pubs_id, data)]
        if not os.path.exists("output"):
            os.mkdir("output")
        with open("output/prediction_embedding.json", "w") as f:
            json.dump(json_prediction, f)
        # config.write_logs(i)


if __name__ == '__main__':
    # rerank_output_prediction()

    rerank_output_prediction()
