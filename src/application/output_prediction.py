import json
import os

from .rerank_for_evaluation import BertRerankPrediction
from .retrieve_for_evaluation import Retrieve


def retrieval_output_prediction(k=5):
    retrieval_model = Retrieve("", "", "bm25", retrieval_kwargs=dict(k=k), transformation_kwargs=dict(add_title=True), prediction=True)
    data = retrieval_model.retrieve()
    pubs_id = retrieval_model.pubs["id"].values
    json_prediction = [{"pub_id":pub_id, "experts": experts} for pub_id, experts in zip(pubs_id, data)]
    if not os.path.exists("output"):
        os.mkdir("output")
    with open("output/prediction.json", "w") as f:
        json.dump(json_prediction, f)

def rerank_output_prediction():
    results = BertRerankPrediction("scibert_24h", k=60).rerank(threshold=0.1057)

    json_prediction = [{"pub_id": pub_id, "experts": experts} for pub_id, experts in results.items()]
    if not os.path.exists("output"):
        os.mkdir("output")
    with open("output/prediction_rerank.json", "w") as f:
        json.dump(json_prediction, f)


if __name__ == '__main__':
    rerank_output_prediction()