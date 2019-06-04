#encoding=utf-8
import numpy as np
import random

def readData(filePath):
    dataList = []
    fr = open(filePath, 'r')
    lines = fr.readlines()
    for line in lines:
        dataList.append(float(line) * 100)
    return dataList
dataList=readData("./ape/train_dev.hter")
bins_temp = 20
begin = 0
end = 100
hist, _ = np.histogram(dataList, bins=bins_temp, range=(begin, end))
samlpe_number = 100000
hist=hist/len(dataList)
dataList_1=readData("./ape/500K.hter")
hist_number = hist * samlpe_number

# print(sum(hist))
# print(len(dataList))
qujian = np.linspace(begin, end, bins_temp + 1, endpoint=True)

cls=[[] for v in range(len(qujian)-1) ]

for i, value in enumerate(dataList_1):
    for j in range(len(qujian) - 1):
        if value >= qujian[j] and value < qujian[j + 1]:
            cls[j].append(i)

res=[]
for ii,cls_temp in enumerate(cls):
    samlpe_temp=min(len(cls_temp),int(round(hist_number[ii])))
    res_temp=random.sample(cls_temp,samlpe_temp)
    # print(res_temp)
    res.extend(res_temp)
# print(res)
# print(dataList_1[70796])
def filterFile(file,filter_file,res1):
    f = open(file, 'r',encoding='utf-8')
    w = open(filter_file, 'w',encoding='utf-8')
    lines = f.readlines()
    for i in res1:
     w.write(lines[i])

filterFile("./ape/500K.src",'./ape/500K_filter.src',res)
filterFile("./ape/500K.mt",'./ape/500K_filter.mt',res)
filterFile("./ape/500K.hter",'./ape/500K_filter.hter',res)