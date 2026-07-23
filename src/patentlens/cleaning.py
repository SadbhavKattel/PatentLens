"""Text cleaning for patent titles/abstracts, shared by the training notebook and the Streamlit app."""

import re

import nltk
from nltk.corpus import stopwords

PATENT_STOPWORDS = {
    'device', 'devices', 'method', 'methods', 'apparatus', 'apparatuses',
    'system', 'systems', 'configured', 'plurality', 'embodiment',
    'embodiments', 'disclosed', 'disclosure', 'provided', 'comprising',
    'comprise', 'comprises', 'invention', 'present', 'said', 'including',
    'include', 'includes', 'included', 'according', 'wherein', 'thereof',
    'therein', 'thereby', 'may', 'can', 'one', 'first', 'second', 'third',
    'least', 'example', 'examples', 'aspect', 'aspects', 'unit', 'units',
    'portion', 'portions', 'having', 'based', 'associated', 'using',
    'used', 'use', 'also', 'further', 'more', 'thus', 'via',
}


def get_stopwords() -> set:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english')) | PATENT_STOPWORDS


def clean_text(text, stop_words: set) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def _split_pipe_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x:
        return []
    return x.split('|')


def prepare_dataframe(df):
    """Clean titles/abstracts and derive the columns the retrievers/evaluation need.

    Idempotent and side-effect-free: returns a new DataFrame, leaves the input untouched.
    """
    stop_words = get_stopwords()
    df = df.copy()

    df['cpc_codes'] = df['cpc_codes'].apply(_split_pipe_list)
    df['cited_patents'] = df['cited_patents'].apply(_split_pipe_list)

    df['clean_title'] = df['title'].apply(lambda t: clean_text(t, stop_words))
    df['clean_abstract'] = df['abstract'].apply(lambda t: clean_text(t, stop_words))
    df['clean_text'] = (df['clean_title'] + ' ' + df['clean_abstract']).str.strip()

    df['primary_cpc'] = df['cpc_codes'].apply(lambda codes: codes[0] if codes else None)
    df['primary_cpc_class'] = df['primary_cpc'].apply(lambda c: c[:4] if c else None)
    df['clean_word_count'] = df['clean_text'].apply(lambda t: len(t.split()))

    return df
