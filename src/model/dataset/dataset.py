from torch.utils.data import Dataset
import random


class InfoNCEDataset(Dataset):
    def __init__(self, paper_text: List[str], user_texts: List[List[str]], hard_negatives: List[List[str]] = None,
                 num_of_neg=4, num_of_hard_neg=1):
        assert len(paper_text) == len(user_texts)
        self.paper_text = paper_text
        self.user_texts = user_texts
        self.hard_negatives = hard_negatives
        self.num_of_neg = num_of_neg
        self.num_of_hard_neg = num_of_hard_neg
        # breakpoint()
        self.full_user_text = list(itertools.chain.from_iterable(user_texts))
        self.full_paper_data = [paper_text[i] for i in range(len(paper_text)) for user_index in range(len(user_texts[i]))]
        self.full_paper_index = [i for i in range(len(paper_text)) for user_index in range(len(user_texts[i]))]

    @staticmethod
    def generate_negative_samples(pos_text, full_text, num_to_generate):
        neg_samples = random.sample(full_text, num_to_generate + 15)
        return list(filter(lambda x: x not in pos_text, neg_samples))[:num_to_generate]

    def __len__(self):
        return len(self.full_user_text)

    def __getitem__(self, idx):
        return InputExample(texts=[self.full_paper_data[idx],
                                    self.full_user_text[idx],
                                    *generate_negative_samples(self.user_texts[self.full_paper_index[idx]], self.full_user_text, self.num_of_neg),
                                    *random.sample(self.hard_negatives[self.full_paper_index[idx]], self.num_of_hard_neg)],
                                 label=self.full_paper_index[idx])