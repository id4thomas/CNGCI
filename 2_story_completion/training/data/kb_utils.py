from tqdm import tqdm


def atomic_apply_template(row):
    """
    columns: ['event', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant', 'prefix', 'split']
    oEffect: (Then,) PersonY ~
    oReact: PersonY feels ~
    oWant: (Then,) PersonY wants (to) ~ / PersonY hopes (to) ~
    xAttr: PersonX is ~
    xEffect: (Then,) PersonX ~
    xIntent: PersonX was intended (to) ~
    xNeed: (Before this,) PersonX had (to) ~ / PersonX needed (to) ~
    xReact: PersonX feels ~
    xWant: (Then,) PersonX wants (to) ~ / PersonX hopes (to) ~
    """
    template = {
        "oEffect": "PersonY ",
        "oReact": "PersonY feels ",
        "oWant": "PersonY wants to ",
        "xAttr": "PersonX is ",
        "xEffect": "PersonX ",
        "xIntent": "PersonX was intended to ",
        "xNeed": "PersonX had to ",
        "xReact": "PersonX feels ",
        "xWant": "PersonX wants to ",
    }
    sentences = []
    for col, tails in row.items():
        if col == "event":
            continue
        for tail in tails:
            if tail == "none":
                continue
            if tail.lower().startswith("to "):
                tail = tail[3:]
            tail = template[col] + tail
            sentences.append((row["event"], tail, col))
    return sentences


def conceptnet_apply_template(conceptnet_data):
    template = {
        "RelatedTo": lambda x, y: f"{x} is related to {y}",
        "IsA": lambda x, y: f"{x} is {y}",
        "PartOf": lambda x, y: f"{x} is a part of {y}",
        "HasA": lambda x, y: f"{y} belongs to {x}",
        "UsedFor": lambda x, y: f"{x} is used for {y}",
        "CapableOf": lambda x, y: f"{x} can {y}",
        "AtLocation": lambda x, y: f"{x} is at {y}",
        "Causes": lambda x, y: f"{x} causes {y}",
        "HasSubevent": lambda x, y: f"{y} is also {x}",
        "HasFirstSubevent": lambda x, y: f"{y} happens first before {x}",
        "HasLastSubevent": lambda x, y: f"{y} happens lastly after {x}",
        "HasPrerequisite": lambda x, y: f"{y} needs to be happened before {x}",
        "HasProperty": lambda x, y: f"{x} has {y} as a property",
        "MotivatedByGoal": lambda x, y: f"Someone may {x} for {y}",
        "ObstructedBy": lambda x, y: f"{x} can be prevented by {y}",
        "Desires": lambda x, y: f"{x} desires {y}",
        "CreatedBy": lambda x, y: f"{y} creates {x}",
        "Synonym": lambda x, y: f"{x} is similar to {y}",
        "Antonym": lambda x, y: f"{x} is opposite to {y}",
        "DistinctFrom": lambda x, y: f"{x} is distinct from {y}",
        "DerivedFrom": lambda x, y: f"{x} is {y}",
        "SymbolOf": lambda x, y: f"{x} may represents {y}",
        "DefinedAs": lambda x, y: f"{y} can be defined as {x}",
        "MannerOf": lambda x, y: f"{x} is way to {y}",
        "LocatedNear": lambda x, y: f"{x} is usually located near to {y}",
        "HasContext": lambda x, y: f"{x} is about {y}",
        "SimilarTo": lambda x, y: f"{x} is similar to {y}",
        "EtymologicallyRelatedTo": lambda x, y: f"{x} and {y} have a common origin",
        "EtymologicallyDerivedFrom": lambda x, y: f"{x} is derived from {y}",
        "CausesDesire": lambda x, y: f"{x} makes someone want {y}",
        "MadeOf": lambda x, y: f"{x} is made of {y}",
        "ReceivesAction": lambda x, y: f"{x} can be done to {y}",
    }
    sentences = []
    for item in tqdm(conceptnet_data, total=conceptnet_data.num_rows):
        if item["lang"] != "en":
            continue

        rel = item["rel"].split("/")[-1]
        if rel not in template:
            continue

        if item["sentence"] is not None and item["sentence"] != "":
            sentences.append(
                (item["sentence"].replace("[[", "").replace("]]", ""), rel)
            )
            continue

        arg1 = item["arg1"].split("/")[-1].replace("_", " ")
        arg2 = item["arg2"].split("/")[-1].replace("_", " ")
        sentences.append((template[rel](arg1, arg2), rel))

    return sentences
