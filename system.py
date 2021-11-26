from utils import vectorizeDocs, getDocs, getQueries, saveToFile, bulkQuery
from index import Index
from document import Document
from query import Query


def main():
    DOC_PATH = "./resources/Trec_microblog11.txt"
    QUERY_PATH = "./resources/topics_MB1-49.txt"
    RESULT_PATH = "./results/Results.txt"

    index = Index()
    docs = getDocs(DOC_PATH)

    index.bulkIndex(docs)

    docVecLens = vectorizeDocs(docs)

    queries = getQueries(QUERY_PATH)

    results = bulkQuery(queries, index, docVecLens)

    saveToFile(results, RESULT_PATH)


if __name__ == '__main__':
    main()
