#using classes
import numpy as np
import sys
from io import StringIO


with open(sys.argv[1], 'r') as f:
    train = f.read()

d_train=np.genfromtxt(StringIO(train),delimiter='',skip_header=1,dtype='str')

with open(sys.argv[2], 'r') as f:
    test = f.read()
d_test=np.genfromtxt(StringIO(test),delimiter='',skip_header=1,dtype='str')

nmax=int(sys.argv[3])     


class decisiontree:
    def __init__(self,index,predict=0):
        self.left = None
        self.right = None
        self.index=index
        self.predict=predict

def gini_gain(data,split):
    temp=data[:,split]
    last=data[:,-1]
    unt=np.unique(temp)
    unl=np.unique(last)
    if len(unt)==2 and len(unl)==2:
        c1=c2=c3=c4=0
        n1=n2=0
        for i in range(len(temp)):
            if temp[i]==unt[0]:
                n1=n1+1
                if last[i]==unl[0]:
                    c1=c1+1
                else:
                    c2=c2+1
            elif temp[i]==unt[1]:
                n2=n2+1
                if last[i]==unl[0]:
                    c3=c3+1
                else:
                    c4=c4+1
#            y1=c1+c2
#            y2=c3+c4
        m1=m2=0
        for i in range(len(last)):
            if last[i]==unl[0]:
                m1=m1+1
            if last[i]==unl[1]:
                m2=m2+1
        d=1-(m1/len(temp))**2-(m2/len(temp))**2
        gi=(n1)/len(temp)*(1-(c1/n1)**2-(c2/n1)**2)+(n2)/len(temp)*(1-(c3/n2)**2-(c4/n2)**2)
        gini=d-gi      
        return gini
    else:
        gini=0
        return gini
      
def node(data):
    blank=np.array([])
    for i in range(data.shape[1]-1):
        blank=np.append(blank,gini_gain(data,i))
    g_max=np.max(blank)
    arg=np.argwhere(blank==g_max)
    return g_max,int(arg[0])

def split(data):
    arg1=arg2=np.array([])
    un=np.unique(data[:,node(data)[1]])
    if len(un)==2:
        for i in range(len(data)):
            if data[i,node(data)[1]]==un[0]:
                arg1=np.append(arg1,i)
            else:
                arg2=np.append(arg2,i)
        f1=f2=np.zeros(data.shape[1])
        for i in range(len(arg1)):
            f1=np.vstack([f1,data[int(arg1[i])]])
        for i in range(len(arg2)):
            f2=np.vstack([f2,data[int(arg2  [i])]])
        f1=f1[1:,:]
        f2=f2[1:,:]
        #print(f1.shape)
        #print(f2.shape)
        return f1,f2
    else:
        return None,None

def majority_vote(data):
    t=data
    un=np.unique(t)
    if len(un)==2:
        c1=c2=0
        for j in range(len(t)):
            if t[j]==un[0]:
                c1=c1+1
            else:
                c2=c2+1
        if c1>c2:
            for j in range(len(t)):
                t[j]=un[0]
        elif c2>c1:
            for j in range(len(t)):
                t[j]=un[1]  
        elif c1==c2:
            a=np.sort(un)
            for j in range(len(t)):
                t[j]=a[-1]
#            print(np.unique(t),'2')
        return np.unique(t)
    else:
        for j in range(len(t)):
            t[j]=un[0]
#            print(np.unique(t),'1')
        return np.unique(t)

def recurse(data,n):
    #print(n)
    index = node(data)[1]
    root = decisiontree(index)
    if len(data)!=0:
        if n==nmax:
            predict= majority_vote(data[:,-1])
            root.predict = predict
            #print(root.predict,'pred')
        else:
#                n=n+1
            if node(data)[0]==0:
                root.predict = majority_vote(data[:,-1])
                #print(root.predict,'pred')
            else:
                l,r=split(data)
                #print(root.node(data)[1],'split')
                root.left = recurse(l,n+1)

                root.right = recurse(r,n+1)
    return root
    
def predict(data,tree):
    d=data[:,:-1]
    #print(root.left)
    un=np.unique(data[:,tree.index])
    #print(un)
    return [_predict(inputs,un,tree) for inputs in d]
    
def _predict(inputs,un,root):
    node = root
    while node.left:
        if inputs[node.index] ==un[0]:
            node=node.left
        elif inputs[node.index] ==un[1]:
            node=node.right
    return node.predict

tree = recurse(d_train,0)

test = np.array(predict(d_test,tree))

train=np.array(predict(d_train,tree))

num=0
for i in range(len(d_test)):
    if d_test[i,-1]==test[i]:
        num=num+1

num1=0
for i in range(len(d_train)):
    if d_train[i,-1]==train[i]:
        num1=num1+1

err_test=1-num/len(d_test)
err_train=1-num1/len(d_train)

f1=open(f'{sys.argv[4]}',"w")
for i in range(len(train)):
    f1.write(str(train[i][0])+"\n")
f1.close()

f2=open(f'{sys.argv[5]}',"w")
for i in range(len(test)):
    f2.write(str(test[i][0])+"\n")
f2.close()

f3=open(f'{sys.argv[6]}',"w")

f3.write(f'error(train): {err_train}\nerror(test): {err_test}')
f3.close()
