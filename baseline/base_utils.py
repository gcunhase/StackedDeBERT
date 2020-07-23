
INTENTION_TAGS = {
    'ChatbotCorpus': {'0': 'DepartureTime',
                      '1': 'FindConnection'}
}

INTENTION_TAGS_WITH_SPACE = {
    'ChatbotCorpus': {'0': 'DepartureTime',
                      '1': 'FindConnection'}
}


SENTIMENT_TAGS = {'sentiment140': {'0': 'Negative',
                                   '1': 'Positive'},
                  }

LABELS_ARRAY_INT = {
    "chatbotcorpus": [0, 1],
    "sentiment140": [0, 1],
}

LABELS_ARRAY = {
    "chatbotcorpus": ["0", "1"],
    "sentiment140": ["0", "1"],
}


def get_label(dataset_name, intent_name, dict_type="intention"):
    if dict_type == "intention":
        tags = INTENTION_TAGS[dataset_name]
    else:
        tags = SENTIMENT_TAGS[dataset_name]
    for k, v in tags.items():
        if intent_name.lower() in v.lower():
            return int(k)
    return -1
