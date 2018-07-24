from normalization import parse_document, normalize_corpus, normalize_document
from feature_extractors import build_feature_matrix


def get_corpus(name):

    corpus = []
    with open(name, 'r') as corpus_file:
        for doc in corpus_file:
            if doc.strip():
                corpus.append(parse_document(doc))
        corpus_file.close()
        return corpus


def search(corpus):
    """Search Demo function"""

    normalized_corpus = normalize_corpus(corpus)
    vectorizer, feature_matrix = build_feature_matrix(
        normalized_corpus, 'tfidf')

    q = input('Enter search query. Press "Enter" to stop: \n')

    while q != '':
        q = normalize_document(q)
        q_tfidf = vectorizer.transform([q])
        ans_mat = q_tfidf.dot(feature_matrix.transpose())
        ans_list = []
        for j in range(ans_mat.shape[1]):
            if ans_mat[0, j] > 0:
                ans_list.append((j, ans_mat[0, j]))
        ans_list.sort(key=lambda x: x[1], reverse=True)

        print()
        print('************ {} ************'.format(q))

        for item in ans_list[:5]:
            print()
            print('Document no. {}, rank: {}'.format(item[0], item[1]))
            print(corpus[item[0]][:])
            print()
            print()

        q = input('Enter search query. Press "Enter" to stop: \n')
    print()


corpus = get_corpus('corpus400.txt')

search(corpus)
