from typing import List


def convert_keywords_to_string(keywords: list, replace_symbol=" "):
    if type(keywords) != list:
        return " "
    try:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * i["w"]) for i in keywords])

    except KeyError:
        return " ".join([" ".join([i["t"].replace(" ", replace_symbol)] * 1) for i in keywords])


def base_data_transformation(full_data_dataset, keywords_transform=False, **kwargs):
    replace_key = "_" if keywords_transform else " "
    prfs_keywords = full_data_dataset["tags"].apply(lambda x: convert_keywords_to_string(x, replace_key))
    prfs_interests = full_data_dataset["interests"].apply(lambda x: convert_keywords_to_string(x, replace_key))
    prfs_full_keywords = prfs_keywords + prfs_interests
    return prfs_full_keywords


def paper_data_transformation(pub, keywords_transform=False, add_abstract=False, add_title=False, **kwargs) -> List[str]:
    replace_key = "_" if keywords_transform else " "
    results = pub["keywords"].apply(lambda x: " ".join(i.replace(" ", replace_key) for i in x)) + pub["title"]
    if add_title:
        results += pub["title"]

    if add_abstract:
        results += pub["abstract"]

    return results

