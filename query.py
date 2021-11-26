class Query:
    def __init__(self, id: str, tokens: [str]):
        self.id = id
        self.tokens = tokens
        self.maxFreq = 0
        for uToken in set(tokens):
            curr = tokens.count(uToken)
            if (curr > self.maxFreq):
                self.maxFreq = curr

    def __repr__(self):
        return f"(Query Num: {self.id}, Tokens: {self.tokens})"
