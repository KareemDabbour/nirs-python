class Document:
    def __init__(self, id: str, tokens: [str]):

        self.id = id
        self.tokens = tokens
        self.uniqueTokens = set(tokens)

    def __repr__(self):
        return f"DocId: {self.id}, Tokens: {self.tokens}"
