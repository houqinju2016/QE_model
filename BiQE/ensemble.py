#encoding=utf-8
import sys
# from src.main_1 import train_ensemble
def ensemble(filelist,file):
    flist=[open(f,'r') for f in filelist]
    lineslist=[f.readlines() for f in flist ]

    fw=open(file,'w')
    # s= sum([0.5328, 0.5381, 0.539, 0.5329, 0.5477, 0.541])
    # w=[i/s for i in [0.5328,0.5381,0.539,0.5329,0.5477,0.541]]
    w=[0.7,0.2,0.1]
    score = [0 for i in range(len(lineslist[0]))]
    for j,l in enumerate(lineslist):
        for i,s in enumerate(l):
            score[i] += w[j]*float(s)

    for a in score:
        fw.write(str('%.6f' % float(a))+'\n')
if __name__ == '__main__':
    flist=[
           #"./en2de2019_result/QE_en2de2019_test_predict_qe_batch8.hter",
           # "./en2de2019_result/QE_en2de2019_test_predict_qe_batch16.hter",
           #"./en2de2019_result/QE_en2de2019_test_predict_qe_batch24.hter",
           #"./en2de2019_result/QE_en2de2019_dev_predict_qe_bt_batch8.hter",
           "./en2de2019_result/QE_en2de2019_test_predict_qe_bt_batch16.hter",
           "./en2de2019_result/QE_en2de2019_test_predict_qe_bt_batch24.hter",
           "./en2de2019_result/QE_en2de2019_test_predict_nmt2mt_bert_batch16.hter",
           #"./en2de2019_result/QE_en2de2019_test_predict_nmt2mt_bert_batch32.hter"
           ]
    ensemble(flist,"./en2de2019_result/test-4.0.7-0.2-0.1.pre")
    # train_ensemble()
