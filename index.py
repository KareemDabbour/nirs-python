# pylint: disable-msg=E0611
from math import log
from document import Document


class Index:
    def __init__(self):
        self.index = {}
        self.numDocs = 0

    def indexDoc(self, doc: Document) -> None:
        for uToken in doc.uniqueTokens:
            tf = doc.tokens.count(uToken)
            entry = (doc.id, 1 + log(tf) if tf > 0 else 0)
            if (not self.index.get(uToken)):
                self.index[uToken] = []
            self.index[uToken].append(entry)
        self.numDocs += 1

    def bulkIndex(self, docs: [Document]) -> None:
        for doc in docs:
            self.indexDoc(doc)

    def getDocFreq(self, term: str) -> int:
        ret = 0
        if (self.index.get(term)):
            ret = len(self.index[term])
        return ret

    def get(self, term: str) -> [(str, float)]:
        return self.index[term]

    def getIDF(self, term: str) -> float:
        ret = 0.0
        if (self.getDocFreq(term) != 0):
            ret = log(self.numDocs/self.getDocFreq(term))
        return ret

    def __repr__(self):
        return f"{self.index}"
