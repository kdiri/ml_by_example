import matplotlib.pyplot as plt
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def letters_only(astr):
    return astr.isalpha()


def main():
    cv = CountVectorizer(stop_words="english", max_features=500)
    groups = fetch_20newsgroups()
    cleaned: list = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()

    for post in groups.data:
        cleaned.append(
            " ".join(
                [
                    lemmatizer.lemmatize(word=word.lower())
                    for word in post.split()
                    if letters_only(word) and word not in all_names
                ]
            )
        )
    transformed = cv.fit_transform(cleaned)
    km = KMeans(n_clusters=20)
    km.fit(transformed)
    labels = groups.target
    plt.scatter(labels, km.labels_)
    plt.xlabel("Newsgroup")
    plt.ylabel("Cluster")
    plt.show()


if __name__ == "__main__":
    main()
