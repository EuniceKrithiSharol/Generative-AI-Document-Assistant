import streamlit as st
import re

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Generative AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)


# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -------------------------------------------------
# SENTENCE SPLITTING
# -------------------------------------------------

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [

        sentence.strip()

        for sentence in sentences

        if len(
            sentence.strip()
        ) > 10
    ]

    return sentences


# -------------------------------------------------
# DOCUMENT SUMMARY
# -------------------------------------------------

def generate_summary(
    text,
    num_sentences=5
):

    sentences = split_sentences(
        text
    )


    if len(sentences) <= num_sentences:

        return " ".join(
            sentences
        )


    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    matrix = vectorizer.fit_transform(
        sentences
    )


    sentence_scores = matrix.sum(
        axis=1
    )


    ranked_sentences = sorted(

        range(
            len(sentences)
        ),

        key=lambda i:

        sentence_scores[
            i,
            0
        ],

        reverse=True
    )


    selected_indices = sorted(

        ranked_sentences[
            :num_sentences
        ]
    )


    summary = [

        sentences[i]

        for i in selected_indices
    ]


    return " ".join(
        summary
    )


# -------------------------------------------------
# KEYWORD EXTRACTION
# -------------------------------------------------

def extract_keywords(
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


    keyword_scores = sorted(

        zip(
            keywords,
            scores
        ),

        key=lambda x:

        x[1],

        reverse=True
    )


    return keyword_scores


# -------------------------------------------------
# QUESTION ANSWERING
# -------------------------------------------------

def answer_question(
    document,
    question
):

    sentences = split_sentences(
        document
    )


    if len(sentences) == 0:

        return (
            "No document content available."
        )


    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    combined_text = sentences + [

        question
    ]


    vectors = vectorizer.fit_transform(
        combined_text
    )


    question_vector = vectors[-1]


    sentence_vectors = vectors[:-1]


    similarities = cosine_similarity(

        question_vector,

        sentence_vectors
    ).flatten()


    best_index = similarities.argmax()


    best_score = similarities[
        best_index
    ]


    if best_score < 0.05:

        return (
            "I could not find a clear answer "
            "to that question in the document."
        )


    return sentences[
        best_index
    ]


# -------------------------------------------------
# SAMPLE DOCUMENT
# -------------------------------------------------

sample_document = """

Artificial Intelligence is transforming modern industries.

Machine Learning enables computer systems to learn
patterns from data and make predictions without being
explicitly programmed for every task.

Natural Language Processing allows computers to
understand and analyze human language.

Generative AI can create new content such as text,
images, code, and other forms of digital information.

Businesses are increasingly using Artificial Intelligence
for automation, customer service, predictive analytics,
fraud detection, recommendation systems, and decision
support.

As Artificial Intelligence continues to evolve, skills in
Machine Learning, Data Science, Natural Language
Processing, and Generative AI are becoming increasingly
important.

"""


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title(
    "🤖 Generative AI Document Assistant"
)


st.markdown(
    "Analyze documents, generate summaries, extract "
    "keywords, and ask questions about your content."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header(
    "🧠 AI Capabilities"
)


st.sidebar.info(
    """
    • Document summarization

    • Keyword extraction

    • Question answering

    • Content analysis

    • AI-style document assistance
    """
)


# -------------------------------------------------
# DOCUMENT INPUT
# -------------------------------------------------

st.header(
    "📄 Add Document Content"
)


document_text = st.text_area(

    "Paste your document or text here",

    value=sample_document,

    height=300
)


document_text = clean_text(
    document_text
)


# -------------------------------------------------
# DOCUMENT METRICS
# -------------------------------------------------

if document_text:

    word_count = len(

        document_text.split()
    )


    sentences = split_sentences(

        document_text
    )


    sentence_count = len(
        sentences
    )


    character_count = len(
        document_text
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Words",
        word_count
    )


    col2.metric(
        "Sentences",
        sentence_count
    )


    col3.metric(
        "Characters",
        character_count
    )


# -------------------------------------------------
# DOCUMENT SUMMARY
# -------------------------------------------------

st.divider()


st.header(
    "📝 AI Document Summary"
)


summary_length = st.slider(

    "Number of important sentences",

    min_value=1,

    max_value=10,

    value=5
)


if st.button(
    "✨ Generate Summary"
):

    if len(document_text) < 20:

        st.warning(
            "Please enter more document content."
        )

    else:

        summary = generate_summary(

            document_text,

            summary_length
        )


        st.success(
            "Summary generated successfully."
        )


        st.write(
            summary
        )


# -------------------------------------------------
# KEYWORD EXTRACTION
# -------------------------------------------------

st.divider()


st.header(
    "🔑 Extract Important Keywords"
)


keyword_count = st.slider(

    "Number of keywords",

    min_value=5,

    max_value=20,

    value=10
)


if st.button(
    "🔍 Extract Keywords"
):

    if len(document_text) < 20:

        st.warning(
            "Please enter more document content."
        )

    else:

        keywords = extract_keywords(

            document_text,

            keyword_count
        )


        keyword_words = [

            keyword[0]

            for keyword in keywords
        ]


        st.subheader(
            "Important Topics"
        )


        st.write(
            ", ".join(
                keyword_words
            )
        )


        st.subheader(
            "Keyword Scores"
        )


        for keyword, score in keywords:

            st.write(

                f"**{keyword.title()}** — "

                f"{score:.3f}"
            )


# -------------------------------------------------
# DOCUMENT QUESTION ANSWERING
# -------------------------------------------------

st.divider()


st.header(
    "💬 Ask Questions About the Document"
)


question = st.text_input(

    "Ask a question about the document",

    placeholder=(
        "Example: What is Generative AI?"
    )
)


if st.button(
    "🤖 Ask AI"
):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    elif len(document_text) < 20:

        st.warning(
            "Please enter document content first."
        )

    else:

        answer = answer_question(

            document_text,

            question
        )


        st.subheader(
            "AI Assistant Response"
        )


        st.success(
            answer
        )


# -------------------------------------------------
# DOCUMENT OVERVIEW
# -------------------------------------------------

st.divider()


st.header(
    "📊 Document Overview"
)


if document_text:

    sentences = split_sentences(

        document_text
    )


    st.subheader(
        "Document Sentences"
    )


    for index, sentence in enumerate(

        sentences,

        start=1
    ):

        st.write(

            f"**{index}.** "

            f"{sentence}"
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()


st.caption(
    "Generative AI Document Assistant | "
    "Python • NLP • Text Processing • "
    "Document Analysis • AI Applications"
)
