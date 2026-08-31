import re

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_document(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def split_document_sentences(text):

    sentences = re.split(

        r"(?<=[.!?])\s+",

        text
    )


    return [

        sentence.strip()

        for sentence in sentences

        if len(
            sentence.strip()
        ) > 10
    ]


def extract_document_keywords(
    text,
    top_n=10
):

    vectorizer = TfidfVectorizer(

        stop_words="english",

        max_features=top_n
    )


    matrix = vectorizer.fit_transform(

        [text]
    )


    keywords = vectorizer.get_feature_names_out()


    scores = matrix.toarray()[0]


    results = sorted(

        zip(
            keywords,
            scores
        ),

        key=lambda x:

        x[1],

        reverse=True
    )


    return results
