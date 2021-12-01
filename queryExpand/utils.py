import re
import string
import time
from typing import Dict, List
from index import Index
from math import log, sqrt
from xml.etree import ElementTree
import gensim.downloader as api

GENSIM_MODEL = "glove-twitter-200"
STOP_WORD_PATH = "../resources/StopWords.txt"
URL_REGEX = re.compile(
    "((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)")


def processStopWords(path):
    ret = []
    with open(path) as f:
        for line in f:
            ret.append(line.rstrip())
    return set(ret)


stopwords_set = processStopWords(STOP_WORD_PATH)


def tokenizeStr(docString: str) -> List[str]:
    return [word for word in preprocStr(docString).split(' ') if word not in stopwords_set and word.strip()]


def preprocStr(docString: str) -> str:
    docString = docString.lower().strip().replace(
        "’", " ").replace("…", " ").replace("“", ' ')
    docString = URL_REGEX.sub('', docString)
    docString = docString\
        .translate(str.maketrans(string.punctuation, ' '*len(string.punctuation), string.digits))
    return docString


def vectorizeDocs(docs: Dict[str, List[str]]) -> Dict[str, float]:
    ret = {}
    for docId, tokens in docs.items():
        docLen: float = 0.0
        for term in set(tokens):
            termFreq = tokens.count(term)
            wTf = 1 + log(termFreq)
            docLen += wTf ** 2
        ret[docId] = sqrt(docLen)
    return ret


def makeQuery(queryTokens: List[str], index: Index, docVecLens: Dict[str, float]) -> Dict[str, float]:
    docRankMap = {}
    queryVecLen = 0.0
    maxFreq = getMaxFreq(queryTokens)
    for term in set(queryTokens):
        queryTfIdf = index.getIDF(
            term) * (queryTokens.count(term) / maxFreq)
        queryVecLen += queryTfIdf ** 2
        for docId, termFreq in index.get(term):
            toBeAdded = queryTfIdf * termFreq
            if docRankMap.get(docId):
                toBeAdded += docRankMap[docId]
            docRankMap[docId] = toBeAdded
    queryVecLen = sqrt(queryVecLen)
    normalize(docRankMap, docVecLens, queryVecLen)
    return dict(sorted(docRankMap.items(), key=lambda item: item[1], reverse=True)[: 1000])


def getMaxFreq(tokens):
    return tokens.count(max(set(tokens), key=tokens.count))


def bulkQuery(queries: Dict[int, List[str]], index: Index, docVecLens: Dict[str, float]) -> List[Dict[str, float]]:
    ret = []
    for query in queries.values():
        ret.append(makeQuery(query, index, docVecLens))
    return ret


def normalize(docRankMap: Dict[str, float], docVecLens: Dict[str, float], queryVecLen: float) -> None:
    for key in docRankMap:
        docRankMap[key] /= (queryVecLen * docVecLens[key])


def getDocs(path: str) -> Dict[str, str]:
    docs = {}
    with open(path) as file:
        for line in file:
            line = line.rstrip()
            docId, text = line.split('\t')
            docs[docId] = preprocStr(text)
    return docs


def tokenizeDocs(docsDict: Dict[str, str]) -> Dict[str, List[str]]:
    ret = {}
    for docId, text in docsDict.items():
        ret[docId] = tokenizeStr(text)
    return ret


def getQueries(path: str) -> Dict[str, str]:
    ret = {}

    queryTree = ElementTree.parse(path)
    qNumRe = re.compile("\d{3}")
    for query in queryTree.getroot():
        qNum = int(qNumRe.search(query[0].text).group(0))
        ret[qNum] = query[1].text
    return ret


def tokenizeQueries(queries: Dict[str, str]):
    ret = {}
    for qId, text in queries.items():
        ret[qId] = tokenizeStr(text)
    return ret


def queryExpand(queries: Dict[str, List[str]]):
    threshold = 0.55
    mult = 1
    start = time.perf_counter()
    print(f"Starting to load model")
    model = api.load(GENSIM_MODEL)
    print(f"Loaded model. Took: {time.perf_counter() - start} seconds")
    for qText in list(queries.values()):
        sim = []
        for text in set(qText):
            if not model.has_index_for(text):
                continue
            simList = [word for (word, val) in model.most_similar(
                text, topn=mult) if val > threshold]
            sim.extend(tokenizeStr(" ".join(simList)))
        qText.extend(sim)


def saveToFile(results: List[Dict[str, float]], path: str) -> None:
    with open(path, 'w') as f:
        for i, result in enumerate(results):
            for rank, (key, value) in enumerate(result.items()):
                line = f"{i+1}\tQ0\t{key}\t{rank+1}\t{value}\tmyRun\n"
                f.write(line)
