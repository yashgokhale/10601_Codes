import numpy as np
import sys
from io import StringIO

with open(sys.argv[1], 'r') as f:
    train = f.read()
d_train=np.genfromtxt(StringIO(train),delimiter=',')
im_tr=d_train[:,0]
att_tr=d_train[:,1:]
em=''

#-----

with open(sys.argv[2], 'r') as f:
    test= f.read()
d_test=np.genfromtxt(StringIO(test),delimiter=',')
im_te=d_test[:,0]
att_te=d_test[:,1:]

#-----

nat=len(att_te[0])
nb=10
n_epochs=int(sys.argv[6])
hun=int(sys.argv[7])
flag=int(sys.argv[8])
lr=float(sys.argv[9])

#-----

labels_tr=np.zeros((len(im_tr),nb))
for i in range(len(labels_tr)):
    labels_tr[i,int(im_tr[i])]=1
labels_te=np.zeros((len(im_te),nb))
for i in range(len(labels_te)):
    labels_te[i,int(im_te[i])]=1

#-----
    
def sigmoid(x):
    ss=1/(1+np.exp(-x))
    return ss    

def softmax(x):
    u=np.exp(x)
    return np.exp(x)/(np.sum(u))

def add_int(x):
    p=np.zeros(nat+1)
    for i in range(len(x)):
        u=np.append(np.array([1]),x[i])
        p=np.vstack([p,u])
    p=p[1:,:]
    return p

def Forward(x,y,w,b,labels):
    a=x@w.T
    z=sigmoid(a)
    z=np.append(np.array([1]),z)
    bb=z@b.T
    yh=softmax(bb)
    j=np.sum(-labels*np.log(yh))
    o=[a,z,bb,yh,j]
    return o

def Backward(x,y,w,b,o,label,hun):
    a,z,bb,yh,j=o
    gyh=yh-label
    dbdbb=z
    gyh=np.reshape(gyh,(nb,1))
    dbdbb=np.reshape(dbdbb,(hun+1,1))
    gb=dbdbb@gyh.T
    djdzz=gyh.T@b[:,1:]
    dzda=sigmoid(a)*(1-sigmoid(a))
    dzda=np.reshape(dzda,(hun,1))
    dadw=x
    dadw=np.reshape(dadw,(len(x),1))
    gw=dadw@(dzda.T*djdzz)
    return gw.T,gb.T

def sgd(X1,Y1,rate,n_epochs,flag,labels,X2,Y2,labels2,hun):
    global em
    if flag==2:
        w=np.zeros((hun,nat+1))
        b=np.zeros((nb,hun+1))
    elif flag==1:
#        np.random.RandomState(1)
        w=np.random.uniform(-0.1,0.1,hun*(nat+1))
        w=w.reshape(hun,nat+1)
        b=np.random.uniform(-0.1,0.1,nb*(hun+1))
        b=b.reshape(nb,hun+1)
        w[:,0]=0
        b[:,0]=0    
    for e in range(n_epochs):
        jj=np.array([])
        jj2=np.array([])
        print(f'Epoch no {e+1}----------')
        for i in range(len(X1)):
            o=Forward(X1[i],Y1[i],w,b,labels[i])
            gw,gb=Backward(X1[i],Y1[i],w,b,o,labels[i],hun)
            w=w-rate*gw
            b=b-rate*gb
        for kk in range(len(X1)):
            o22=Forward(X1[kk],Y1[kk],w,b,labels[kk])
            jj=np.append(jj,o22[-1])       
        for k in range(len(X2)):
            o2=Forward(X2[k],Y2[k],w,b,labels2[k])
            jj2=np.append(jj2,o2[-1])
        em=em+f'epoch={e+1} crossentropy(train): {np.average(jj)}\n'
        em=em+f'epoch={e+1} crossentropy(test): {np.average(jj2)}\n'
        if e==n_epochs-1:
            print(np.average(jj),np.average(jj2))
    return w,b

def predict(x,w,b):
    a=x@w.T
    z=sigmoid(a)
    z=np.append(np.array([1]),z)
    bb=z@b.T
    yh=softmax(bb)
    p=np.argmax(yh)
    return p

#-----

xtr=add_int(att_tr)

xte=add_int(att_te)

wp,bp=sgd(xtr,att_tr,lr,n_epochs,flag,labels_tr,xte,att_te,labels_te,hun)

#-----

pr1=np.array([])
for i in range(len(xtr)):
    temp=predict(xtr[i],wp,bp)
    pr1=np.append(pr1,temp)
    
pr2=np.array([])
for i in range(len(xte)):
    temp=predict(xte[i],wp,bp)
    pr2=np.append(pr2,temp)

n1=n2=0
for i in range(len(pr1)):
    if pr1[i]!=im_tr[i]:
        n1=n1+1
err_tr=n1/len(pr1)

for i in range(len(pr2)):
    if pr2[i]!=im_te[i]:
        n2=n2+1
        
err_te=n2/len(pr2)

em=em+f'error(train): {err_tr}\nerror(test): {err_te}'

#-----

f2=open(f'{sys.argv[3]}',"w")
for i in range(len(pr1)):
    f2.write(str(int(pr1[i]))+"\n")
f2.close()

f2=open(f'{sys.argv[4]}',"w")
for i in range(len(pr2)):
    f2.write(str(int(pr2[i]))+"\n")
f2.close()

f1=open(sys.argv[5],"w")
f1.write(em)
f1.close()

#-----end
#neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt 2 4 2 0.1
