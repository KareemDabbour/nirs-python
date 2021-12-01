from utils import *
from index import Index
import time


def main():
    DOC_PATH = "../resources/Trec_microblog11.txt"
    QUERY_PATH = "../resources/topics_MB1-49.txt"
    RESULT_PATH = "../results/Results.txt"

    start = time.perf_counter()
    index = Index()
    regDocs = getDocs(DOC_PATH)
    docs = tokenizeDocs(regDocs)

    index.bulkIndex(docs)

    docVecLens = vectorizeDocs(docs)

    regQueries = getQueries(QUERY_PATH)

    queries = tokenizeQueries(regQueries)

    results = bulkQuery(queries, index, docVecLens)

    saveToFile(results, RESULT_PATH)
    print(f"Time to finish entire thing: {time.perf_counter() - start}")


if __name__ == '__main__':
    main()
