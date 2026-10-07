import functools
import random
import time


def retry(times=3, base=0.2, exc=(OSError,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            for i in range(times):
                try:
                    return fn(*a, **kw)
                except exc:
                    if i == times - 1:
                        raise
                    time.sleep(base * 2 ** i + random.random() / 10)
        return wrapper
    return deco
