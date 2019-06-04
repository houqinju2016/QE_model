# encoding=utf-8
import sys
import numpy


def hter_format(file, outfile):
    f = open(file, 'r', encoding='utf-8')
    fw = open(outfile, 'w', encoding='utf-8')
    lines = f.readlines()
    res = [0 for ii in range(len(lines))]
    method_name = "njunlp"
    index_1 = 0
    vals = numpy.array([float(val.strip()) for val in lines])
    ind = vals.argsort()
    index_2 = 0
    for r in ind:
        res[r]=index_1
        index_1 += 1

    index_3=0
    for val in lines:
        index_2 += 1
        val = val.strip()
        fw.write(method_name + "\t" + str(index_2) + "\t" + str('%.6f' % float(val)) + "\t" + str(res[index_3]) + "\n")
        index_3+=1
    f.close()
    fw.close()


if __name__ == '__main__':
    hter_format(sys.argv[1], sys.argv[2])
