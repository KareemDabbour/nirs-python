from utils import *
from index import Index
from document import Document
from query import Query
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def main():
    DOC_PATH = "./resources/Trec_microblog11.txt"
    QUERY_PATH = "./resources/topics_MB1-49.txt"
    RESULT_PATH = "./results/Results.txt"
    STOP_WORD_PATH = "./resources/StopWords.txt"

    index = Index()
    regDocs = getDocs(DOC_PATH)
    docs = tokenizeDocs(regDocs)
    # pDocs = getDocsPd(DOC_PATH)

    # cv = CountVectorizer(stop_words="english")
    # cv_matrix = cv.fit_transform(pDocs['text'])
    # df = pd.DataFrame(cv_matrix.toarray(
    # ), index=pDocs['docId'].values, columns=cv.get_feature_names_out())

    # pQs = getQueriesPd(QUERY_PATH)

    # pqMatrix = cv.transform(pQs['text'])
    # df1 = pd.DataFrame(pqMatrix.toarray(
    # ), index=pQs['queryId'].values, columns=cv.get_feature_names_out())

    # print(df.head())
    # print(df1.head())
    index.bulkIndex(docs)

    docVecLens = vectorizeDocs(docs)

    regQueries = getQueries(QUERY_PATH)

    queries = tokenizeQueries(regQueries)

    results = bulkQuery(queries, index, docVecLens)

    # reRank(results, regDocs, regQueries)  # regDocs, regQueries)

    saveToFile(results, RESULT_PATH)


if __name__ == '__main__':
    main()
