"""
    Week 8 assignment
    Pavlidis Pavlos aid20010
    To run: python topic_modelling.py
"""
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import nltk
nltk.download('brown')
from nltk.corpus import brown


def fetch_data():
    data = []
    for fileid in brown.fileids():
        document = ' '.join(brown.words(fileid))
        data.append(document)
    return data


def print_topics(model, vectorizer):
    top_n = 10
    for topic_id, topic in enumerate(model.components_):
        print("-"*10, "Topic %d:" % (topic_id + 1), "-"*10)
        tmp = [(vectorizer.get_feature_names()[i], round(topic[i], 3))
               for i in topic.argsort()[:-top_n - 1:-1]]
        print(tmp, '\n')


def generate_svd_model(tf_idf, tf_idf_docs):
    svd = TruncatedSVD(n_components=15)
    svd_topic_vectors = svd.fit_transform(tf_idf_docs.values)
    print_topics(svd, tf_idf)
    return svd_topic_vectors, svd


def generate_lda_model(tf_idf, tf_idf_docs):
    lda = LatentDirichletAllocation(n_components=15, learning_method='batch')
    lda_topic_docs = lda.fit_transform(tf_idf_docs.values)
    print_topics(lda, tf_idf)
    return lda_topic_docs, lda


def main():
    data = fetch_data()
    token = RegexpTokenizer(r'[a-zA-Z\-][a-zA-Z\-]{2,}')
    tf_idf = TfidfVectorizer(lowercase=True, stop_words='english', tokenizer=token.tokenize)
    tf_idf_docs = tf_idf.fit_transform(data).toarray()
    tf_idf_docs = pd.DataFrame(tf_idf_docs)

    # generate svd model
    print("-"*50, 'SVD METHOD', '-'*50, '\n')
    svd_topic_vectors, svd_model = generate_svd_model(tf_idf, tf_idf_docs)

    # generate lda model
    print("\n", "-"*50, 'LDA METHOD', '-'*50, '\n')
    lda_topic_vectors, lda__model = generate_lda_model(tf_idf, tf_idf_docs)

    x = 1


if __name__ == '__main__':
    main()
