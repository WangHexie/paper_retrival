import itertools
from typing import List
import math


def convert_keywords_to_string(keywords: list, replace_symbol=" ", log_transform=False):
    if type(keywords) != list:
        return " "

    if log_transform:
        transform_func = lambda x: int(math.log(abs(x)+1) + 1)
    else:
        transform_func = lambda x: x

    try:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * transform_func(i["w"])) for i in keywords])

    except KeyError:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * 1) for i in keywords])


def base_data_transformation(full_data_dataset, keywords_transform=False, log_transform=False, add_paper_keywords=False, add_prsf_interest=True,**kwargs):
    replace_key = "_" if keywords_transform else " "
    prfs_keywords = full_data_dataset["tags"].apply(lambda x: convert_keywords_to_string(x, replace_key, log_transform))
    prfs_interests = full_data_dataset["interests"].apply(
        lambda x: convert_keywords_to_string(x, replace_key, log_transform))

    if add_prsf_interest:
        prfs_full_keywords = prfs_keywords + prfs_interests
    else:
        prfs_full_keywords = None

    if add_paper_keywords:
        paper_keywords = full_data_dataset["pub_info"].map(lambda x: " ".join(itertools.chain.from_iterable([i["keywords"] for i in x])))
        prfs_full_keywords = prfs_full_keywords + paper_keywords if prfs_full_keywords is not None else paper_keywords

    if not add_paper_keywords and not add_prsf_interest:
        prfs_full_keywords = prfs_interests

    return prfs_full_keywords


def paper_data_transformation(pub, keywords_transform=False, add_abstract=False, add_title=False, abstract_length=200,
                              **kwargs) -> List[str]:
    replace_key = "_" if keywords_transform else " "
    results = pub["keywords"].apply(lambda x: " ".join(i.replace(" ", replace_key) for i in x))
    if add_title:
        results += pub["title"]

    if add_abstract:
        results += pub["abstract"].apply(lambda x: " ".join(x.split(" ")[:abstract_length]))

    return results


def get_all_pub_info(pub, abstract_length=200, **kwargs):
    keywords = pub["keywords"].apply(lambda x: " ".join(i for i in x))

    return keywords.values.tolist(), pub["title"].values.tolist(), pub["abstract"].apply(lambda x: " ".join(x.split(" ")[:abstract_length])).values.tolist()


def paper_embedding_transformation(keywords, titles, abstracts, weight=None, **kwargs):
    if weight is None:
        weight = [1, 0, 0]
    return sum([keywords*weight[0], titles*weight[1], abstracts*weight[2]])/len(weight)
