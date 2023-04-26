class Value:
    def __init__(self, data: float):
        self.data = data

    def __repr__(self) -> str:
        return f"Value({self.data})"
