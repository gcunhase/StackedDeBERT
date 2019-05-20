
INTENTION_TAGS = {
    'snips': {'0': 'AddToPlaylist',
              '1': 'BookRestaurant',
              '2': 'GetWeather',
              '3': 'PlayMusic',
              '4': 'RateBook',
              '5': 'SearchCreativeWork',
              '6': 'SearchScreeningEvent'},
    'ChatbotCorpus': {'0': 'DepartureTime',
                      '1': 'FindConnection'},
    'AskUbuntuCorpus': {'0': 'MakeUpdate',
                        '1': 'SetupPrinter',
                        '2': 'ShutdownComputer',
                        '3': 'SoftwareRecommendation',
                        '4': 'None'},
    'WebApplicationsCorpus': {'0': 'ChangePassword',
                              '1': 'DeleteAccount',
                              '2': 'DownloadVideo',
                              '3': 'ExportData',
                              '4': 'FilterSpam',
                              '5': 'FindAlternative',
                              '6': 'SyncAccounts',
                              '7': 'None'}
}

INTENTION_TAGS_WITH_SPACE = {
    'snips': {'0': 'AddToPlaylist',
              '1': 'BookRestaurant',
              '2': 'GetWeather',
              '3': 'PlayMusic',
              '4': 'RateBook',
              '5': 'SearchCreativeWork',
              '6': 'SearchScreeningEvent'},
    'ChatbotCorpus': {'0': 'DepartureTime',
                      '1': 'FindConnection'},
    'AskUbuntuCorpus': {'0': 'Make Update',
                        '1': 'Setup Printer',
                        '2': 'Shutdown Computer',
                        '3': 'Software Recommendation',
                        '4': 'None'},
    'WebApplicationsCorpus': {'0': 'Change Password',
                              '1': 'Delete Account',
                              '2': 'Download Video',
                              '3': 'Export Data',
                              '4': 'Filter Spam',
                              '5': 'Find Alternative',
                              '6': 'Sync Accounts',
                              '7': 'None'}
}


LABELS_ARRAY_INT = {
    "snips": [0, 1, 2, 3, 4, 5, 6],
    "chatbotcorpus": [0, 1],
    "askubuntucorpus": [0, 1, 2, 3, 4],
    "webapplicationscorpus": [0, 1, 2, 3, 4, 5, 6, 7],
}

LABELS_ARRAY = {
    "snips": ["0", "1", "2", "3", "4", "5", "6"],
    "chatbotcorpus": ["0", "1"],
    "askubuntucorpus": ["0", "1", "2", "3", "4"],
    "webapplicationscorpus": ["0", "1", "2", "3", "4", "5", "6", "7"],
}


def get_label(dataset_name, intent_name):
    tags = INTENTION_TAGS[dataset_name]
    for k, v in tags.items():
        if intent_name.lower() in v.lower():
            return int(k)
    return -1
