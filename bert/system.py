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

    for (docId, doc), orDoc in zip(list(docs.items())[:5], list(regDocs.values())[:5]):
        print(f"DocId: {docId}\nOriginal Doc: {orDoc} \nTokens: {doc}")
    index.bulkIndex(docs)

    # print(list(index.index.keys())[:100])
    docVecLens = vectorizeDocs(docs)

    regQueries = getQueries(QUERY_PATH)

    queries = tokenizeQueries(regQueries)
    # queryExpand(queries)

    results = bulkQuery(queries, index, docVecLens)

    results = reRank(results, regDocs, regQueries)

    saveToFile(results, RESULT_PATH)
    print(f"Time to finish entire thing: {time.perf_counter() - start}")


if __name__ == '__main__':
    main()
