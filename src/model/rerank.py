import os
from datetime import datetime

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample

from ..config import root_path


class XGBRerank:
    def __init__(self, base_data, paper_data):
        self.paper_data = paper_data
        self.base_data = base_data

    def train(self, features, y):
        pass

    def predict(self, features, y):
        pass

    def evaluate(self, features, y):
        pass


class BertRerank:
    def __init__(self):
        model_args = ClassificationArgs(process_count=1, train_batch_size=96, dataloader_num_workers=1, fp16=False)

        self.model = ClassificationModel("bert", "bert-base-cased", cuda_device=1, args=model_args, use_cuda=True)

    @staticmethod
    def _reformat_example(paper_text, user_text, y):
        assert len(paper_text) == len(user_text) and len(paper_text) == len(y)
        data = [[paper_text[i], user_text[i], y[i]] for i in range(len(y))]
        eval_df = pd.DataFrame(data)
        eval_df.columns = ["text_a", "text_b", "labels"]
        return eval_df

    def train(self, paper_text, user_text, y):
        self.model.train_model(self._reformat_example(paper_text, user_text, y), acc=sklearn.metrics.accuracy_score)
        self.model.save_model("../../../models")
        return self

    def predict(self, paper_text, user_text):
        return self.model.predict([[paper_text[i], user_text[i]] for i in range(len(paper_text))])

    def evaluate(self, paper_text, user_text, y):
        return self.model.eval_model(self._reformat_example(paper_text, user_text, y))


class BertRankSE:
    def __init__(self, batch_size=8, num_epochs=1, model_save_path=os.path.join(root_path, "models"), max_length=512,
                 initial_load=True, model_name='allenai/scibert_scivocab_uncased', device="cuda:1"):
        self.max_length = max_length
        self.model_save_path = model_save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        if initial_load:
            try:
                self.model = CrossEncoder(os.path.join(self.model_save_path, model_name), num_labels=1, device=device,
                                          max_length=max_length)
            except:
                self.model = CrossEncoder(model_name, num_labels=1, device=device, max_length=max_length)
            self.model.save(os.path.join(self.model_save_path, model_name))

    def load_model(self):
        self.model = CrossEncoder(os.path.join(root_path, "models", "manual_save"), device=self.device,
                                  max_length=self.max_length)

        return self

    @staticmethod
    def _reformat_example(paper_text, user_text, labels):
        assert len(paper_text) == len(user_text) and len(paper_text) == len(labels)
        # breakpoint()
        train_examples = [InputExample(texts=[paper, user], label=y) for paper, user, y in
                          zip(paper_text, user_text, labels)]
        return train_examples

    def train(self, paper_text, user_text, y):
        data = self._reformat_example(paper_text, user_text, y)
        train_data, test_data = train_test_split(data, test_size=1000, shuffle=True, random_state=8888)
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

    # def evaluate(self, paper_text, user_text, y):
    #     return self.model.eval_model(self._reformat_example(paper_text, user_text, y))


if __name__ == '__main__':
    a = BertRankSE().load_model().predict(["fdasfa " * 100] * 1000, ["dasfsa "] * 1000)
    print(a)
