from typing import List

from tqdm import tqdm
import numpy as np
from scipy import sparse
from itertools import permutations


class Graph:
    def __init__(self, top_user=10, user_num=5, retrieve_type="linked"):
        self.top_user = top_user
        self.user_num = user_num
        # self.

    def fit(self, base_dataset):
        # init base info
        self.all_users = base_dataset["id"].values
        self.user_id_map = {self.all_users[i]: i for i in range(len(self.all_users))}
        self.id_user_map = {i: self.all_users[i] for i in range(len(self.all_users))}

        #
        all_papers_rough = [i for i in base_dataset["pub_info"].values]

        all_papers = []
        for i in tqdm(all_papers_rough):
            all_papers += i

        all_paper_id = list(map(lambda x: x["id"], all_papers))

        paper_to_user = {id: list() for id in set(all_paper_id)}

        for _, row in tqdm(base_dataset.iterrows()):
            user_id = row.id
            for pub in row.pub_info:
                pub_id = pub["id"]
                paper_to_user[pub_id].append(user_id)

        self.links_in_prfsr = {key: value for key, value in paper_to_user.items() if len(value) > 1}
        self._rank_links()

        return self

    def _rank_links(self):
        shape = (len(self.all_users), len(self.all_users))
        links = [values for _, values in self.links_in_prfsr.items()]

        adj = sparse.lil_matrix(shape)
        for linked_users in links:
            for edge in permutations(linked_users, 2):
                adj[self.user_id_map[edge[0]], self.user_id_map[edge[1]]] += 1

        self.adj = adj

        self.users_close_user = {}
        for i in range(shape[0]):
            row = self.adj[i].tocoo()
            data = list(zip(row.col, row.data))
            sorted_index_value = sorted(data, key=lambda x: x[1], reverse=True)
            self.users_close_user[self.id_user_map[i]] = [self.id_user_map[i[0]] for i in sorted_index_value]

    def retrieve(self, interested_users: List[List[str]]):
        result = []
        for users in interested_users:
            recalled_related_user = []
            for user in users[:self.top_user]:
                recalled_related_user += self.users_close_user[user][:self.user_num]
            result.append(list(set(users+recalled_related_user)))
        return result



if __name__ == '__main__':
    from ..data.dataset import Dataset
    Graph().fit(Dataset().read_base_dataset())
