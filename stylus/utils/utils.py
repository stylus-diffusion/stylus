import time


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, description: str):
        self.description = description
        print(self.description)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f'{self.description}, Elapsed {elapsed}(s).')
