import sys
from collections import defaultdict

import numpy as np
from loguru import logger
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from src.machine_learning.chapter3_spam_detection.read_from_file import (
    main as get_labels,
)
from src.machine_learning.chapter3_spam_detection.constant.email_text import emails_test
from sklearn.model_selection import train_test_split


logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


def letters_only(astr):
    return astr.isalpha()


def clean_text(docs):
    cleaned_docs: list = []
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    for doc in docs:
        cleaned_docs.append(
            " ".join(
                [
                    lemmatizer.lemmatize(word.lower())
                    for word in doc.split()
                    if letters_only(word) and word not in all_names
                ]
            )
        )

    return cleaned_docs


def read_from_file(file_path):
    with open(file_path, "r", encoding="ISO-8859-1") as infile:
        contents = infile.read()
    return contents


def get_label_index(labels):
    """
    Possible return ex: {0: [3000, 3001, 3002, 3003, ...... 6670, 6671], 1: [0, 1, 2, 3, ...., 2998, 2999]}
    :param labels: grouped sample indices by class
    :return: dictionary: with class label as key, corresponding prior as the value
    """
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index


def get_prior(labels: dict):
    """
    Compute prior based on training samples
    :param labels:
    :return:
    """
    prior = {label: len(index) for label, index in labels.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior


def get_likelihood(
    term_document_matrix: csr_matrix, label_index: defaultdict, smoothing: int = 0
):
    """
    Compute likelihood based on training samples
    :param term_document_matrix: sparse matrix
    :param label_index: defaultdict: grouped sample indices by class
    :param smoothing: integer: additive Laplace smoothing
    :return:
    """
    likelihood: dict = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)

    return likelihood


def get_posterior(term_document_matrix, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood.
    :param term_document_matrix: sparse matrix
    :param prior: dictionary: with class label as key, corresponding prior as the value
    :param likelihood: dictionary: with class label as key,
            corresponding conditional probability vector as value
    :return: dictionary, with class label as key, corresponding posterior as value
    """
    num_docs = term_document_matrix.shape[0]
    posteriors: list = []
    for i in range(num_docs):
        # posterior is proportional to prior * likelihood
        # = exp(log(prior * likelihood))
        # = exp(log(prior) + log(likelihood))
        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)

        counts = term_document_vector.data
        indices = term_document_vector.indices

        for count, index in zip(counts, indices):
            posterior[label] += np.log(likelihood_label[index] * count)
        # exp(-1000):exp(-999) will cause zero division error,
        # however it equates to exp(0):exp(1)
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                # if one's log value is excessively large, assign it infinity
                posterior[label] = float("inf")
        # normalize so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float("inf"):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())

    return posteriors


def process():
    smoothing = 1
    cv = CountVectorizer(stop_words="english", max_features=500)
    emails, labels = get_labels()
    cleaned_emails = clean_text(emails_test)
    term_docs = cv.fit_transform(emails)
    term_docs_test = cv.fit_transform(cleaned_emails)
    feature_names = cv.get_feature_names()
    label_index = get_label_index(labels)
    prior = get_prior(label_index)
    likelihood = get_likelihood(term_docs, label_index, smoothing)
    posterior = get_posterior(term_docs_test, prior, likelihood)
    logger.info(likelihood[0][:5])
    logger.info(feature_names[:5])
    logger.info(posterior)

    # x_train, x_test, y_train, y_yest = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
    # logger.info(len(x_train), len(y_train))
    # logger.info(len(x_test), len(y_yest))

if __name__ == "__main__":
    process()
