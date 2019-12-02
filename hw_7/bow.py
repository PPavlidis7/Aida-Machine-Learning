import glob
import itertools
import json
import os
import numpy as np

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_files():
    path = r'dialogues'  # use your path
    all_files = glob.glob(
        os.path.join(path, "*.txt"))  # advisable to use os.path.join as this makes concatenation OS independent
    data = []
    for file in all_files:
        with open(file, 'r') as f:
            data.append(f.readlines())

    for index, values in enumerate(data):
        for inner_index, record in enumerate(values):
            _tmp_value = json.loads(record)
            _tmp_value['turns'] = ' '.join(_tmp_value['turns'])
            data[index][inner_index] = _tmp_value

    data = list(itertools.chain(*data))
    df = pd.DataFrame(data)
    return df


def generate_bows(data):
    # tokenizer to remove unwanted elements from out data like symbols and numbers
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    return cv.fit_transform(data['turns']), cv


def find_best_bows(bows, cv_model, user_input, df_data):
    user_bow = cv_model.transform([user_input, ])
    similarities = cosine_similarity(user_bow, bows).flatten()
    best_similarities = list(similarities.argsort()[:-11:-1])
    print(df_data.iloc[best_similarities, [0, 3]])


def main():
    data = read_files()
    bows, cv_model = generate_bows(data)
    user_input = input("Give me your question\n")
    find_best_bows(bows, cv_model, user_input, data)


if __name__ == '__main__':
    main()
