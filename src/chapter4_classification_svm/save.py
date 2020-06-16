import os
import glob
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

k = 10
k_fold = StratifiedKFold(n_splits=k)
all_names = set(names.words())
emails, labels = [], []
lemmatizer = WordNetLemmatizer()
# To load and label the spam email files to label '1'
spam_file_path = 'datasets/enron1/spam'
for filename in glob.glob(os.path.join(spam_file_path, '*.txt')):
    with open(filename, 'r', encoding='ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(1)

# To load and label the non spam email files to label '0'
ham_file_path = 'datasets/enron1/ham'
for filename in glob.glob(os.path.join(ham_file_path, '*txt')):
    with open(filename, 'r', encoding='ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(0)


def letters_only(astr):
    return astr.isalpha()


def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join(lemmatizer.lemmatize(word.lower()) for word in doc.split()
                                     if letters_only(word) and word not in all_names))
        # lowercase everything, isalpha does number and punc. removal, not in all_names removes words
    return cleaned_docs


cleaned_emails = clean_text(emails)
smoothing_factor_option: list = [1.0, 2.0, 3.0, 4.0, 5.0]
auc_record = defaultdict(float)
for train_indices, test_indices in k_fold.split(cleaned_emails, labels):
    print(cleaned_emails[train_indices], cleaned_emails[test_indices])
    X_train, X_indices = cleaned_emails[train_indices], cleaned_emails[test_indices]
    Y_train, Y_test = labels[train_indices], labels[test_indices]
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english", max_features=8000)
    term_docs_train = tfidf_vectorizer.fit_transform(X_train)
    term_docs_test = tfidf_vectorizer.transform(X_test)
    for smoothing_factor in smoothing_factor_option:
        clf = MultinomialNB(alpha=smoothing_factor, fit_prior=True)
        clf.fit(term_docs_train, Y_train)
        prediction_prob = clf.predict_proba(term_docs_test)
        pos_prob = prediction_prob[:, 1]
        auc = roc_auc_score(Y_test, pos_prob)
        auc_record[smoothing_factor] += auc
print(f"Max features smoothing fit prior auc:")
for smoothing, smoothing_record in auc_record.items():
    print(f"\t 8000 \t {smoothing}\t true {smoothing_record} ")
