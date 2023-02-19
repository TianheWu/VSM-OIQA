


a = []
b = []

with open('./data/IQA_ODI/IQA_ODI_dis_label.txt', 'r') as listFile:
    for line in listFile:
        a.append(line)

with open('./data/IQA_ODI/ref_dis_ID.txt', 'r') as listFile:
    for line in listFile:
        b.append(line[:-1])

with open('./data/IQA_ODI/ref_dis_label.txt','w') as f:
    for i in range(len(a)):
        f.write(str(b[i] + ' ' + a[i]))