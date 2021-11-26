import re
import string
from document import Document
from query import Query
from index import Index
# pylint: disable-msg=E0611
from math import log, sqrt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer
from xml.etree import ElementTree


URL_REGEX = re.compile(
    "((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)")
nltk.download('stopwords')
tknzr = TweetTokenizer()
stopwords_set = set(stopwords.words("english")).union(
    set(stopwords.words("custom")))


def tokenizeStr(docString: str) -> [str]:
    return [word.strip() for word in tknzr.tokenize(preprocStr(docString)) if word not in stopwords_set and len(word.strip()) > 2]


def preprocStr(docString: str) -> str:

    # making all lower case
    docString = docString.lower()
    docString = docString.strip()
    docString = URL_REGEX.sub('', docString)
    docString = docString.translate(str.maketrans('/', ' '))
    # Removing puncuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    docString = docString.translate(str.maketrans('', '', string.punctuation))
    docString = docString.translate(str.maketrans('', '', string.digits))
    return docString


def vectorizeDocs(docs: [Document]) -> {str, float}:
    ret = {}
    for doc in docs:
        docLen: float = 0.0
        for term in doc.uniqueTokens:
            termFreq = doc.tokens.count(term)
            wTf = 1 + log(termFreq)
            docLen += wTf ** 2
        ret[doc.id] = sqrt(docLen)
    return ret


def makeQuery(query: Query, index: Index, docVecLens: {str: float}) -> {str: float}:
    docRankMap = {}
    queryVecLen = 0.0
    for term in set(query.tokens):
        queryTfIdf = index.getIDF(
            term) * (list(set(query.tokens)).count(term) / query.maxFreq)
        queryVecLen += queryTfIdf ** 2
        for docId, termFreq in index.get(term):
            toBeAdded = queryTfIdf * termFreq
            if docRankMap.get(docId):
                toBeAdded += docRankMap[docId]
            docRankMap[docId] = toBeAdded
    queryVecLen = sqrt(queryVecLen)
    normalize(docRankMap, docVecLens, queryVecLen)
    return dict(sorted(docRankMap.items(), key=lambda item: item[1])[: 1000])


def bulkQuery(queries: [Query], index: Index, docVecLens: {str: float}) -> [{str: float}]:
    ret = []
    for query in queries:
        ret.append(makeQuery(query, index, docVecLens))
    return ret


def normalize(docRankMap: {str: float}, docVecLens: {str: float}, queryVecLen: float) -> None:
    for key, value in docRankMap.items():
        docRankMap[key] = value / (queryVecLen * docVecLens[key])


def getDocs(path: str) -> [Document]:
    docs = []
    with open(path) as file:
        for line in file:
            line = line.rstrip()
            docId, text = line.split('\t')
            tokenizedText = tokenizeStr(text)
            docs.append(Document(docId, tokenizedText))
    return docs


def getQueries(path: str) -> [Query]:
    ret = []

    queryTree = ElementTree.parse(path)
    qNumRe = re.compile("\d{3}")
    for query in queryTree.getroot():
        qNum = int(qNumRe.search(query[0].text).group(0))
        qTokens = tokenizeStr(query[1].text)
        expandQuery(qTokens)
        ret.append(
            Query(qNum, qTokens))
    return ret


def expandQuery(tokens: [str]) -> None:
    pass


def saveToFile(results: [{str: float}], path: str) -> None:
    with open(path, 'w') as f:
        for i, result in enumerate(results):
            for rank, (key, value) in enumerate(result.items()):
                line = f"{i+1}\tQ0\t{key}\t{rank+1}\t{value}\tmyRun\n"
                f.write(line)
