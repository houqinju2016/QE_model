# encoding=utf-8
import sys


def addlabel_1(inputFile, outputFile):
    tran = open(inputFile, 'r')
    new_tran = open(outputFile, 'w')

    lines = tran.readlines()
    index = 0
    for line in lines:
        new_tran.write(line.strip() + ' ' + '(' + str(index) + ')')
        index += 1


def addlabel_4(inputFile, outputFile):
    tran = open(inputFile, 'r')
    new_tran = open(outputFile, 'w')

    lines = tran.readlines()
    index = 0
    for i, line in enumerate(lines):

        if (i != 0 and i % 4 == 0):
            index += 1
        new_tran.write(line.strip() + ' ' + '(' + str(index) + ')'+"\n")


def merge_file(filelist, mergefile):
    f = [open(file, 'r') for file in filelist]
    fw = open(mergefile, 'w')
    lines0 = f[0].readlines()
    lines1 = f[1].readlines()
    lines2 = f[2].readlines()
    lines3 = f[3].readlines()
    for a, b, c, d in zip(lines0, lines1, lines2, lines3):
        fw.write(a.strip() + '\n')
        fw.write(b.strip() + '\n')
        fw.write(c.strip() + '\n')
        fw.write(d.strip() + '\n')


if __name__ == '__main__':
    filelist=["test.1","test.2","test.3","test.4"]
    merge_file(filelist,"test.m.txt")
    addlabel_4("test.m.txt","test.m.label.txt")