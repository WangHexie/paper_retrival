import math
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import numpy as np

import os

from .persist import PersistModel
from ..config import root_path
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist


def load_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name,
                                      cache_dir=os.path.join(root_path, "models", model_name.split("/")[-1]),
                                      proxies={"https": '172.16.12.110:5687', "http": '172.16.12.110:5687'})
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=os.path.join(root_path, "models", model_name.split("/")[-1]),
                                              proxies={"https": '172.16.12.110:5687', "http": '172.16.12.110:5687'})
    return model, tokenizer


def output_best_match_document(query_embedding, document_embedding, article, number_to_output=20, output_index=False):
    """

    :param output_index:
    :param query_embedding: 2D, (1, embedding_size)
    :param document_embedding: (n, embedding_size)
    :param article: array(n)
    :param number_to_output:
    :return:
    """
    dis = cdist(document_embedding, query_embedding, metric='cosine')
    dis_index = np.argsort(dis.flatten())

    if not output_index:
        return article[dis_index[:number_to_output]], None
    else:
        return article[dis_index[:number_to_output]], dis_index[:number_to_output]


class BatchBuilder(Dataset):
    def __init__(self, data, batch_size, device="cuda:1"):
        self.data = data
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        print("test batch builder, batch_size:", self.batch_size)
        print("test batch builder, data_shape:", self.data[list(self.data.keys())[0]].shape)
        return math.ceil(len(self.data[list(self.data.keys())[0]]) / self.batch_size)  # round up

    def __getitem__(self, index):
        return {i: self.data[i][index * self.batch_size:(index + 1) * self.batch_size] for i in self.data.keys()}


class Embedding:
    def __init__(self, model_name, device="cuda", batch_size=32, half_float=True):
        """
        convert_name_to_path("最终模型_3.1")
        :param model_name:
        """
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.encoder = self.model
        self.half_float = half_float

        if self.half_float:
            self.encoder = self.encoder.half()

        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

    # def load_model(self):
    #     self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)
    #     self.encoder = self.model
    #     if self.half_float:
    #         self.encoder = self.encoder.half()
    #     return self

    def get_embedding(self, sentences: List[str]):
        torch.set_grad_enabled(False)

        self.encoder.to(self.device)

        token_result = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding="longest",
                                                        truncation=True)

        a = BatchBuilder(token_result, self.batch_size, self.device)

        full_embedding = []
        for i in tqdm(range(len(a))):
            tokens_input = a[i]
            tokens_input = {key: tokens_input[key].to(self.device) for key in tokens_input.keys()}

            embeddings = self.encoder(**tokens_input)[0]  # dont know why return a array
            masked_embedding = embeddings * tokens_input["attention_mask"].unsqueeze(-1)
            embedding_sum = masked_embedding.sum(1)
            masked_sum = tokens_input["attention_mask"].sum(1)
            final_embedding = embedding_sum / masked_sum.unsqueeze(-1)

            full_embedding.append(final_embedding.detach().cpu().numpy())

        return np.concatenate(full_embedding)


class PersistEmbeddingModel:
    def __init__(self, cache_name, emd_model_kwargs):
        self.cache_name = cache_name
        self.emd_model_kwargs = emd_model_kwargs

    def load(self, document: List[str]):
        pm = PersistModel(self.cache_name)
        if pm.exist():
            return pm.load()
        else:
            emb = Embedding(**self.emd_model_kwargs)
            embedding = emb.get_embedding(document)
            pm.save(embedding)
            return embedding


if __name__ == '__main__':
    m, t = load_model_and_tokenizer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
    # tokenizer = download_auto_model("sentence-transformers/paraphrase-TinyBERT-L6-v2")
    # model = download_auto_model("sentence-transformers/paraphrase-TinyBERT-L6-v2")
