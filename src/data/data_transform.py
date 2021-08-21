import itertools
from typing import List
import math
import functools
import pandas as pd


def convert_keywords_to_string(keywords: list, replace_symbol=" ", log_transform=False):
    if type(keywords) != list:
        return " "

    if log_transform:
        transform_func = lambda x: int(math.log(abs(x) + 1) + 1)
    else:
        transform_func = lambda x: x

    try:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * transform_func(i["w"])) for i in keywords])

    except KeyError:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * 1) for i in keywords])


def create_list_of_string_from_base_data(full_data_dataset, keywords_transform=False, log_transform=False,
                                         add_paper_keywords=False,
                                         add_prsf_interest=True, add_paper_title=False, **kwargs):
    replace_key = "_" if keywords_transform else " "
    prfs_keywords = full_data_dataset["tags"].apply(
        lambda x: "[TAG] " + convert_keywords_to_string(x, replace_key, log_transform))
    prfs_interests = full_data_dataset["interests"].apply(
        lambda x: "[INTEREST] " + convert_keywords_to_string(x, replace_key, log_transform))
    columns_name = []

    string_to_merge = []
    if add_prsf_interest:
        columns_name.append("prfs_interests")

        string_to_merge.append(prfs_interests)
        columns_name.append("prfs_keywords")

        string_to_merge.append(prfs_keywords)

    if add_paper_title:
        paper_title = full_data_dataset["pub_info"].map(lambda x: "[TITLE] " + " ".join([i["title"] for i in x]))
        string_to_merge.append(paper_title)
        columns_name.append("pub_title")

    if add_paper_keywords:
        paper_keywords = full_data_dataset["pub_info"].map(
            lambda x: "[KEYWORD] " + " ".join(itertools.chain.from_iterable([i["keywords"] for i in x])))
        string_to_merge.append(paper_keywords)

        columns_name.append("pub_keywords")

    if len(string_to_merge) == 0:
        string_to_merge.append(prfs_interests)
        columns_name.append("prfs_interests")

    return string_to_merge, columns_name


def base_data_transformation(full_data_dataset, keywords_transform=False, log_transform=False, add_paper_keywords=False,
                             add_prsf_interest=True, add_paper_title=False, **kwargs):
    string_to_merge, _ = create_list_of_string_from_base_data(full_data_dataset, keywords_transform, log_transform,
                                                              add_paper_keywords,
                                                              add_prsf_interest, add_paper_title, **kwargs)

    prfs_full_keywords = functools.reduce(lambda x, y: x + y, string_to_merge)
    return "[USER] " + prfs_full_keywords


def base_data_transformation_to_diction_records(full_data_dataset, keywords_transform=False, log_transform=False,
                                                add_paper_keywords=False, add_prsf_interest=True, add_paper_title=False,
                                                **kwargs):
    string_to_merge, columns_name = create_list_of_string_from_base_data(full_data_dataset, keywords_transform,
                                                                         log_transform,
                                                                         add_paper_keywords,
                                                                         add_prsf_interest, add_paper_title, **kwargs)

    data_df = pd.DataFrame(string_to_merge).T
    data_df.columns = columns_name
    return data_df.to_dict('records')


def paper_data_transformation(pub, keywords_transform=False, add_abstract=False, add_title=False, abstract_length=200,
                              **kwargs) -> List[str]:
    replace_key = "_" if keywords_transform else " "
    results = pub["keywords"].apply(lambda x: "[KEYWORD] " + " ".join(i.replace(" ", replace_key) for i in x))
    if add_title:
        results += "[TITLE]" + pub["title"]

    if add_abstract:
        results += pub["abstract"].apply(lambda x: "[ABSTRACT] " + " ".join(x.split(" ")[:abstract_length]))

    return "[PAPER] " + results


def get_all_pub_info(pub, abstract_length=200, **kwargs):
    keywords = pub["keywords"].apply(lambda x: " ".join(i for i in x))

    return keywords.values.tolist(), pub["title"].values.tolist(), pub["abstract"].apply(
        lambda x: " ".join(x.split(" ")[:abstract_length])).values.tolist()


def paper_embedding_transformation(keywords, titles, abstracts, weight=None, **kwargs):
    if weight is None:
        weight = [1, 0, 0]
    return sum([keywords * weight[0], titles * weight[1], abstracts * weight[2]]) / len(weight)
