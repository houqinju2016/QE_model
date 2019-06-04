import sys


def paste(file1, file2, file3, file4):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    f3 = open(file3, 'r')
    f4 = open(file4, 'w')
    for a, b, c in zip(f1.readlines(), f2.readlines(), f3.readlines()):
        a = a.strip()
        b = b.strip()
        c = c.strip()
        f4.write(a + '\t' + b + '\t' + c + '\n')


if __name__ == '__main__':
    paste(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
