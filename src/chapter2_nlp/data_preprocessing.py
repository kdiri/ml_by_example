import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def letters_only(astr):
    return astr.isalpha()


def main():
    cleaned: list = []
    cv = CountVectorizer(stop_words="english", max_features=500)
    groups = fetch_20newsgroups()
    all_names = np.unique(names.words())  # set(names.words())
    lemmatizer = WordNetLemmatizer()
    for post in groups.data:
        cleaned.append(
            " ".join(
                [
                    lemmatizer.lemmatize(word.lower())
                    for word in post.split()
                    if letters_only(word) and word not in all_names
                ]
            )
        )
    transformed = cv.fit_transform(cleaned)
    print(cv.get_feature_names())


if __name__ == "__main__":
    main()
