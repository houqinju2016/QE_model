# encoding=utf-8
import sys


def cul_precision(goldenFile, predictFile):
    f_golden = open(goldenFile, 'r')
    f_predict = open(predictFile, 'r')

    lines_golden = f_golden.readlines()
    lines_predict = f_predict.readlines()

    index = 0

    num_pre = 0
    num_tot = 0

    bad_count = 0
    good_count = 0
    for g, p in zip(lines_golden, lines_predict):
        index += 1
        list_g = g.strip().split()
        list_p = p.strip().split()

        length_g = len(list_g)
        length_p = len(list_p)

        if (length_g > length_p):
            print(index)
            bad_count += 1
        if (length_g ==length_p):
            good_count += 1


        for word_g, word_p in zip(list_g, list_p):
            if (word_g == word_p):
                num_pre += 1

        num_tot += length_g
    pre = num_pre / num_tot
    print("bad_count:"+str(bad_count))
    print("good_count:" + str(good_count))
    print(num_pre)
    print(num_tot)
    print(pre)
    return pre


if __name__ == '__main__':
    cul_precision(sys.argv[1], sys.argv[2])
