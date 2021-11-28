# pylint: disable-msg=E0611
from math import log
from typing import List, Dict


class Index:
    def __init__(self):
        self.index = {}
        self.numDocs = 0

    def indexDoc(self, docId: str, docText: List[str]) -> None:
        for uToken in set(docText):
            tf = docText.count(uToken)
            entry = (docId, 1 + log(tf) if tf else 0)
            if (not self.index.get(uToken)):
                self.index[uToken] = []
            self.index[uToken].append(entry)
        self.numDocs += 1

    def bulkIndex(self, docs: Dict[str, List[str]]) -> None:
        for docId, tokens in docs.items():
            self.indexDoc(docId, tokens)

    def getDocFreq(self, term: str) -> int:
        ret = 0
        if (self.index.get(term)):
            ret = len(self.index[term])
        return ret

    def get(self, term: str):
        return self.index[term] if self.index.get(term) else []

    def getIDF(self, term: str) -> float:
        ret = 0.0
        if (self.getDocFreq(term) != 0):
            ret = log(self.numDocs/self.getDocFreq(term))
        return ret

    def __repr__(self):
        return f"{self.index}"
