def preprocess_coreference(document):
    text = document["text"]
    mentions = document["mentions"]

    mentions = sorted(
        mentions,
        key=lambda m: (m["onset"], -(m["offset"] - m["onset"]))
    )

    # Add mention text in place
    for mention in mentions:
        mention["text"] = text[mention["onset"]:mention["offset"]]

    opening_tag = "{___ "
    closing_tag = "}"
    zero_node_sign = "∅"

    characters = list(text)

    for mention in mentions:
        if "type" not in mention:
            mention["type"] = "mention"

    # Process longer mentions first when they share an onset
    non_zero_mentions = [
        m for m in mentions
        if m["type"] != "zero_node"
    ]

    zero_node_mentions = [
        m for m in mentions
        if m["type"] == "zero_node"
    ]

    for mention in non_zero_mentions:
        characters[mention["onset"]] = (
            f"{opening_tag}{characters[mention['onset']]}"
        )
        characters[mention["offset"] - 1] = (
            f"{characters[mention['offset'] - 1]}{closing_tag}"
        )

    for mention in zero_node_mentions:
        characters[mention["offset"] - 1] = (
            f"{characters[mention['offset'] - 1]} "
            f"{opening_tag}{zero_node_sign}{closing_tag}"
        )

    document["input"] = "".join(characters)
    document["mentions"] = mentions

    return document