import os
import sys
import time


def follow(fh, poll=0.25):
    fh.seek(0, os.SEEK_END)
    while True:
        line = fh.readline()
        if not line:
            time.sleep(poll)
            continue
        yield line


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        for ln in follow(fh):
            if "ERROR" in ln:
                sys.stdout.write(ln)
                sys.stdout.flush()
