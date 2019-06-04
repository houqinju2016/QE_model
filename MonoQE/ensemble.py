#encoding=utf-8
import sys
def ensemble(file1,file2,file3):
    f1=open(file1,'r')
    f2=open(file2,'r')
    f3=open(file3,'w')
    lines1=f1.readlines()
    lines2=f2.readlines()
    w1=0.6
    w2=1-w1
    for a,b in zip(lines1,lines2):
        c=w1*float(a)+w2*float(b)
        f3.write(str(c)+'\n')
if __name__ == '__main__':
    ensemble(sys.argv[1], sys.argv[2],sys.argv[3])