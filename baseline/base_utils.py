
INTENTION_TAGS = {
    'snips': {'0': 'AddToPlaylist',
              '1': 'BookRestaurant',
              '2': 'GetWeather',
              '3': 'PlayMusic',
              '4': 'RateBook',
              '5': 'SearchCreativeWork',
              '6': 'SearchScreeningEvent'}
}

INTENTION_TAGS_WITH_SPACE = {
    'snips': {'0': 'AddToPlaylist',
              '1': 'BookRestaurant',
              '2': 'GetWeather',
              '3': 'PlayMusic',
              '4': 'RateBook',
              '5': 'SearchCreativeWork',
              '6': 'SearchScreeningEvent'}
}


SENTIMENT_TAGS = {'sentiment140': {'0': 'Negative',
                                   '1': 'Positive'},
                  }

LABELS_ARRAY_INT = {
    "snips": [0, 1, 2, 3, 4, 5, 6],
    "sentiment140": [0, 1],
}

LABELS_ARRAY = {
    "snips": ["0", "1", "2", "3", "4", "5", "6"],
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
