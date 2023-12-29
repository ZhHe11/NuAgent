from collections import deque


class A:
    def __init__(self) -> None:
        self.cnt = 0
        self.data = [i for i in range(10)]

    def __iter__(self):
        self.cnt += 1
        print("call iter:", self.cnt)
        return self

    def __next__(self):
        print("data", self.data[self.cnt])
        return self.data[self.cnt]


deque(A(), maxlen=0)
