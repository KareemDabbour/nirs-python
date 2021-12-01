from utils import *
from index import Index


def main():
    DOC_PATH = "../resources/Trec_microblog11.txt"
    QUERY_PATH = "../resources/topics_MB1-49.txt"
    RESULT_PATH = "../results/Results.txt"

    start = time.perf_counter()

    print("Starting")
    index = Index()
    regDocs = getDocs(DOC_PATH)
    docs = tokenizeDocs(regDocs)
    print(
        f"Finished tokenizing docs. Took {time.perf_counter() - start} seconds.")
    index.bulkIndex(docs)
    print(f"Number of unique tokens: {len(index.index.keys())}")

    docVecLens = vectorizeDocs(docs)

    regQueries = getQueries(QUERY_PATH)

    queries = tokenizeQueries(regQueries)

    queryExpand(queries)

    print(
        f"Finished expanding queries. Took {time.perf_counter() - start} seconds.")
    results = bulkQuery(queries, index, docVecLens)

    saveToFile(results, RESULT_PATH)

    print(f"Time to complete: {time.perf_counter() - start}")


if __name__ == '__main__':
    main()
