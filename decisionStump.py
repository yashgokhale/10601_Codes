import sys

from io import StringIO

import numpy as np

#------taking in user inputs
with open(sys.argv[1], 'r') as f:
    train = f.read()
    
with open(sys.argv[2], 'r') as f:
    test = f.read()

index=int(sys.argv[3])

#f1=open(f'{sys.argv[4]}',"w")

#f2=open(f'{sys.argv[5]}',"w")

#f3=open(f'{sys.argv[6]}',"w")

#----converting user data to array
d_train=np.genfromtxt(StringIO(train),delimiter='',skip_header=1,dtype='str')

d_test=np.genfromtxt(StringIO(test),delimiter='',skip_header=1,dtype='str')

#----extracting first and last column

tr1=d_train[:,index]
trm=d_train[:,-1]
te1=d_test[:,index]
tem=d_test[:,-1]

arg_1=np.array([])
arg_2=np.array([])

#--create a single for loop for train data
c1=c2=c3=c4=0
un1=np.unique(tr1)
unm=np.unique(trm)

for i in range(len(tr1)):
    if tr1[i]==un1[0]:
        arg_1=np.append(arg_1,i)
        if trm[i]==unm[0]:
            c1=c1+1;
        if trm[i]==unm[1]:
            c2=c2+1;
        if max(c1,c2)==c1:
            for k,j in enumerate(arg_1):
                trm[int(j)]=unm[0]
        elif max(c1,c2)==c2:
            for k,j in enumerate(arg_1):
                trm[int(j)]=unm[1]
    elif tr1[i]==un1[1]:
        arg_2=np.append(arg_2,i)
        if trm[i]==unm[0]:
            c3=c3+1;
        if trm[i]==unm[1]:
            c4=c4+1;
        if max(c3,c4)==c3:
            for k,j in enumerate(arg_2):
                trm[int(j)]=unm[0]
        elif max(c3,c4)==c4:
            for k,j in enumerate(arg_2):
                trm[int(j)]=unm[1] 
                
c_train=0

d_train=np.genfromtxt(StringIO(train),delimiter='',skip_header=1,dtype='str')

for i in range(len(trm)):
    if trm[i]!=d_train[:,-1][i]:
        c_train=c_train+1
err_train=(c_train/len(trm))

#--create a single for loop for test data
c11=c21=c31=c41=0
un11=np.unique(te1)
unm1=np.unique(tem)

arg_11=np.array([])
arg_21=np.array([])

for i in range(len(te1)):
    if te1[i]==un11[0]:
        arg_11=np.append(arg_11,i)
        if tem[i]==unm1[0]:
            c11=c11+1;
        if tem[i]==unm1[1]:
            c21=c21+1;
        if max(c11,c21)==c11:
            for k,j in enumerate(arg_11):
                tem[int(j)]=unm1[0]
        elif max(c11,c21)==c21:
            for k,j in enumerate(arg_11):
                tem[int(j)]=unm1[1]
    elif te1[i]==un11[1]:
        arg_21=np.append(arg_21,i)
        if tem[i]==unm1[0]:
            c31=c31+1;
        if tem[i]==unm1[1]:
            c41=c41+1;
        if max(c31,c41)==c31:
            for k,j in enumerate(arg_21):
                tem[int(j)]=unm1[0]
        elif max(c31,c41)==c41:
            for k,j in enumerate(arg_21):
                tem[int(j)]=unm1[1] 
                
c_test=0

d_test=np.genfromtxt(StringIO(test),delimiter='',skip_header=1,dtype='str')

for i in range(len(tem)):
    if tem[i]!=d_test[:,-1][i]:
        c_test=c_test+1
err_test=(c_test/len(tem))

f1=open(f'{sys.argv[4]}',"w")
for i in range(len(trm)):
    f1.write(str(trm[i])+"\n")
f1.close()

f2=open(f'{sys.argv[5]}',"w")
for i in range(len(tem)):
    f2.write(str(tem[i])+"\n")
f2.close()

f3=open(f'{sys.argv[6]}',"w")

f3.write(f'error(train): {err_train:1.6}\nerror(test): {err_test}')
f3.close()

print(err_train,err_test)
