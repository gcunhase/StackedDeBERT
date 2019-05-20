from collections import defaultdict, OrderedDict
import operator
from sklearn.feature_extraction.text import TfidfVectorizer


S1 = "The car is driven on the road"
S2 = "The truck is driven on the highway"
S3 = "The truck is green"
S = [S1, S2, S3]
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform(S)
feature_names = vectorizer.get_feature_names()

dictionary = defaultdict(lambda: [])
for col in response.nonzero()[1]:
    dictionary[feature_names[col]] = response[0, col]
    # print(feature_names[col], ' - ', response[0, col])

dictionary_sorted = OrderedDict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
for k, v in dictionary_sorted.items():
    print("{} - {}".format(k, v))
