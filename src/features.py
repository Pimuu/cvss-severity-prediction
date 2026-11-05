"""Text feature extraction: TF-IDF configurations.

The main configuration (word 1-2 grams, sublinear TF) is the standard strong
baseline for short technical text. Alternatives are used in the ablation study.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def word_tfidf(ngram_range=(1, 2)):
    """Word n-gram TF-IDF. Lower-cased; tokens keep hyphens/dots-free alphanumerics."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        min_df=2,              # drop hapax n-grams (mostly random product names)
        max_df=0.95,           # drop near-ubiquitous terms
        sublinear_tf=True,     # 1 + log(tf): dampens repeated terms
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",  # ignore pure numbers / version strings
    )


def char_tfidf():
    """Character 3-5 grams within word boundaries (robust to morphology / typos)."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)


def word_plus_char():
    """Concatenation of word 1-2 gram and char 3-5 gram TF-IDF spaces."""
    return FeatureUnion([("word", word_tfidf((1, 2))), ("char", char_tfidf())])


FEATURE_SETS = {
    "word 1-gram": lambda: word_tfidf((1, 1)),
    "word 1-2-gram": lambda: word_tfidf((1, 2)),
    "word 1-3-gram": lambda: word_tfidf((1, 3)),
    "char 3-5-gram": char_tfidf,
    "word 1-2 + char 3-5": word_plus_char,
}
