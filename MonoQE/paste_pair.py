#encoding=utf-8
import sys


def paste(file1, file2, file3):
    f1 = open(file1, 'r',encoding="utf-8")
    f2 = open(file2, 'r',encoding="utf-8")
    f3 = open(file3, 'w',encoding="utf-8")
    for a, b in zip(f1.readlines(), f2.readlines()):
        a = a.strip()
        b = b.strip()

        f3.write(a + ' ||| ' + b + '\n')


if __name__ == '__main__':
    paste(sys.argv[1], sys.argv[2], sys.argv[3])
