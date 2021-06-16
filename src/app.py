import math
import io
import json
import os
import string

import configargparse
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------------------------------------
# Classes


# https://github.com/taki0112/Vector_Similarity
class TS_SS:

    def Cosine(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.dot(vec1, vec2.T)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def VectorSize(self, vec: np.ndarray):
        return np.linalg.norm(vec)

    def Euclidean(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.linalg.norm(vec1-vec2)

    def Theta(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.arccos(self.Cosine(vec1, vec2)) + np.radians(10)

    def Triangle(self, vec1: np.ndarray, vec2: np.ndarray):
        theta = np.radians(self.Theta(vec1, vec2))
        return (self.VectorSize(vec1) * self.VectorSize(vec2) * np.sin(theta))/2

    def Magnitude_Difference(self, vec1: np.ndarray, vec2: np.ndarray):
        return abs(self.VectorSize(vec1) - self.VectorSize(vec2))

    def Sector(self, vec1: np.ndarray, vec2: np.ndarray):
        ED = self.Euclidean(vec1, vec2)
        MD = self.Magnitude_Difference(vec1, vec2)
        theta = self.Theta(vec1, vec2)
        return math.pi * (ED + MD)**2 * theta/360


    def __call__(self, vec1: np.ndarray, vec2: np.ndarray):
        try:
            m = self.Triangle(vec1, vec2) * self.Sector(vec1, vec2)
            v = m.item((0, 0))
            return 0 if np.isnan(v) else v
        except:
            pass

        return 0

# --------------------------------------------------------------------------------
# Methods


def is_interactive():
    return __name__ == '__main__'


def corpus_dir():
    return os.path.join(os.path.dirname(__file__), 'corpus')


def load_corpus(task):
    corpus = []  # The text of the file goes here
    labels = []  # The name of the file goes here

    task_dir = os.path.join(corpus_dir(), task)

    for currentDir, _, files in os.walk(task_dir):
        # Get the absolute path of the currentDir parameter
        currentDir = os.path.abspath(currentDir)

        # Traverse through all files
        for fileName in files:
            fullPath = os.path.join(currentDir, fileName)
            with io.open(fullPath, 'r', encoding='utf-8', errors='ignore') as f:
                contents = f.read()

            if 'orig' not in fileName:
                corpus.append(contents)
                labels.append(fileName)
            else:
                # The original is the first entry
                corpus.insert(0, contents)
                labels.insert(0, fileName)

    return corpus, labels


def load_metadata():
    fileName = os.path.join(corpus_dir(), 'corpus-final09.xls')
    df = pd.read_excel(fileName, index_col=0, sheet_name='File list')
    return df


def filter_tokens(doc):
    for token in doc:
        if (
            not token.is_punct and
            not token.is_space
        ):
            yield token

# --------------------------------------------------------------------------------
# Lambda entry

def run(options):
    # Load the metadata
    mdf = load_metadata()

    # Load the sources
    corpus, labels = load_corpus(options.task)

    # Have spaCy process the documents - with just basic tokenizing
    nlp = spacy.load("en_core_web_sm", exclude=['ner', 'lemmatizer', 'textcat'])
    docs = [
        ' '.join([t.norm_ for t in filter_tokens(doc)])
        for doc in nlp.pipe(corpus)
    ]

    # Prepare a TF-IDF vectorizer that reads sing words and two-word pairs
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(docs)

    # Peek at the raw values
    # df = pd.DataFrame(tfidf_vectors.T.todense(),
    #                   index=tfidf_vectorizer.get_feature_names())
    # print(df)

    # Score similarity
    similarity = TS_SS()
    input_vector = tfidf_vectors[0].todense()
    _scores = [
        similarity(input_vector, tfidf_vectors[i].todense())
        for i in range(1, len(docs))
    ]

    # Create a Pandas series from the scores
    ts_ss_scores = pd.Series(_scores, index=labels[1:], name='difference')

    # Make a data frame
    df = ts_ss_scores.to_frame() #.reset_index().rename(columns={'index': 'File'})

    # Evaluate
    df['plagiarized'] = df['difference'] < 0.000339

    # Merge in the metadata
    df = pd.concat([mdf, df], axis=1, join="inner")
    df.sort_values(by='difference', ascending=True, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 132)
    print(df)

# --------------------------------------------------------------------------------
# Main

def build_arg_parser():
    p = configargparse.ArgParser(prog='plag_detect',
                                 description='determines if text is plagiarized')
    p.add('task', choices=['taska', 'taskb', 'taskc', 'taskd', 'taske'],
          help='the task to analyze')

    return p


if is_interactive():
    # Get the arguments from the command line
    p = build_arg_parser()
    options = p.parse_args()

    # Execute the main process
    run(options)
