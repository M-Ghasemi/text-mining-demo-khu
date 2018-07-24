
from contractions import CONTRACTION_MAP
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
import unicodedata
from nltk.corpus import wordnet as wn


stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def expand_contractions(text, contraction_mapping):

    contractions_pattern = re.compile(
        '({})'.format('|'.join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Annotate text tokens with POS tags


def pos_tag_text(text):

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text


# lemmatize text based on POS tags
def lemmatize_text(text):

    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(
        None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def unescape_html(parser, text):

    return parser.unescape(text)


def normalize_document(doc, lemmatize=True, tokenize=False):
    doc = expand_contractions(doc, CONTRACTION_MAP)

    if lemmatize:
        doc = lemmatize_text(doc)
    else:
        doc = doc.lower()
    doc = remove_special_characters(doc)
    doc = remove_stopwords(doc)
    if tokenize:
        doc = tokenize_text(doc)

    return doc


def normalize_corpus(corpus, lemmatize=True, tokenize=False):

    normalized_corpus = []
    for text in corpus:
        # text = expand_contractions(text, CONTRACTION_MAP)
        # if lemmatize:
        #     text = lemmatize_text(text)
        # else:
        #     text = text.lower()
        # text = remove_special_characters(text)
        # text = remove_stopwords(text)
        # if tokenize:
        #     text = tokenize_text(text)

        normalized_corpus.append(normalize_document(text, lemmatize, tokenize))

    return normalized_corpus


def parse_document(document, sent_tokenize=False):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize(
            'NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]

    return sentences if sent_tokenize else ' '.join(sentences)
