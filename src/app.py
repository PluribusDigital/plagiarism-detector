import math
import io
import json
import os
import string

import boto3
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


def load_corpus():
    corpus = []
    labels = []

    if not is_interactive():
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('jmf-sandbox-2021')
        for i, obj in enumerate(bucket.objects.all()):
            # if 'orig' not in obj.key:
            #     continue

            object = s3.Object(obj.bucket_name, obj.key)
            contents = object.get()['Body'].read().decode('utf-8', errors='ignore')
            corpus.append(contents)
            labels.append(obj.key)
    else:
        for currentDir, _, files in os.walk('/Users/farley/Desktop/plagarism-corpus'):
            # Get the absolute path of the currentDir parameter
            currentDir = os.path.abspath(currentDir)

            # Traverse through all files
            for fileName in files:
                fullPath = os.path.join(currentDir, fileName)
                # if 'orig' not in fileName:
                #     continue

                with io.open(fullPath, 'r', encoding='utf-8', errors='ignore') as f:
                    contents = f.read()
                corpus.append(contents)
                labels.append(fileName)

    return corpus, labels


def filter_tokens(doc):
    for token in doc:
        if (
            not token.is_punct and
            not token.is_space
        ):
            yield token

# --------------------------------------------------------------------------------
# Lambda entry

def root_handler(event, context):
    data = event.get('body', '')

    # Load the sources
    corpus, labels = load_corpus()

    # Add the data as a document
    corpus.append(data)
    labels.append('input')

    nlp = spacy.load("en_core_web_sm", exclude=['ner', 'lemmatizer', 'textcat'])
    docs = [
        ' '.join([t.norm_ for t in filter_tokens(doc)])
        for doc in nlp.pipe(corpus)
    ]

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_vectors = tfidf_vectorizer.fit_transform(docs)

    # Peek at the raw values
    # df = pd.DataFrame(tfidf_vectors.T.todense(),
    #                   index=tfidf_vectorizer.get_feature_names())
    # print(df)

    # Score similarity
    similarity = TS_SS()
    input_vector = tfidf_vectors[-1].todense()
    _scores = [
        similarity(input_vector, tfidf_vectors[i].todense())
        for i in range(len(docs) - 1)
    ]
    ts_ss_scores = pd.Series(_scores, index=labels[0:-1], name='difference')

    # Make a data frame
    df = ts_ss_scores.to_frame().reset_index().rename(columns={'index': 'File'})

    # Evaluate
    df['plagiarized_by_input'] = df['difference'] < 0.000225
    df['related_to_input'] = df['difference'] < 0.000333
    df['matches_input'] = df['difference'] == 0.0
    df.sort_values(by='difference', ascending=True, inplace=True)

    # output
    results = df.loc[df['related_to_input'] == True].to_dict(orient='records')

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(results)
    }

# --------------------------------------------------------------------------------
# Main


if is_interactive():
    body = '''
In mathematics and computer science, dynamic programming is a methodology of the solution of the problems that exhibit the properties of overlapping subproblems and optimal substructure (described below). The methodology takes much less time rather than naive methods.
The term was originally used during the 1940s by Richard Bellman to describe the process of solving problems where one needs to find the best decisions one after another. By 1953, he had refined this to the modern meaning. The field was founded as a systems analysis and engineering topic that is recognized by the IEEE. Bellman's contribution is remembered in the name of the Bellman equation, a central result of dynamic programmer, which restates an optimization problem in recursive form.
The word "programming" in "dynamic programming" has no particular connection to computer programming in general , and instead of this it comes from the term "mathematical programming", a synonym for optimization. Therefore, the "program" is the optimal plan for action that is produced. For example, a finalized schedule of events at an exhibition is sometimes called a program.
Optimal substructure means that optimal solutions of subproblems can be used to find the optimal solutions of the overall problem. For instance, the shortest path to a goal from a vertex in a graph can be found by first computing the shortest path to the goal from all adjacent vertices. After this, it is using this to pick the best overall path. In a word, we can solve a problem with optimal substructure using a three-step process.
    '''
    result = root_handler({'body': body}, {})
    print(json.dumps(result, indent=2))
