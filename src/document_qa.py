from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import re


def split_sentences(text):

    sentences = re.split(

        r"(?<=[.!?])\s+",

        text
    )


    return [

        sentence.strip()

        for sentence in sentences

        if sentence.strip()
    ]


def find_answer(
    document,
    question
):

    sentences = split_sentences(
        document
    )


    if not sentences:

        return (
            "No document content available."
        )


    vectorizer = TfidfVectorizer(

        stop_words="english"
    )


    texts = sentences + [

        question
    ]


    vectors = vectorizer.fit_transform(
        texts
    )


    question_vector = vectors[-1]


    document_vectors = vectors[:-1]


    similarity_scores = cosine_similarity(

        question_vector,

        document_vectors
    ).flatten()


    best_match = similarity_scores.argmax()


    score = similarity_scores[
        best_match
    ]


    if score < 0.05:

        return (
            "No relevant answer was found "
            "in the document."
        )


    return sentences[
        best_match
    ]
