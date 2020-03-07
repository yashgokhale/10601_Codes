#practice code
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

arg1=np.array([])
for i in range(len(d_train)):
    arg1=np.append(arg1,i)
d_train=np.column_stack([d_train,arg1])

arg2=np.array([])
for i in range(len(d_test)):
    arg2=np.append(arg2,i)
d_test=np.column_stack([d_test,arg2])

d_t2=d_train

def gini_gain(data,split):
    n1=n2=0
    c1=c2=c3=c4=0
    last=data[:,-2]
    un_l=np.unique(last)
    temp=data[:,split]
    un_t=np.unique(temp)
    if len(un_t)==2:
        if len(un_l)==2:
            for i in range(len(temp)):
                if temp[i]==un_t[0]:
                    n1=n1+1
                    if last[i]==un_l[0]:
                        c1=c1+1
                    if last[i]==un_l[1]:
                        c2=c2+1
                if temp[i]==un_t[1]:
                    n2=n2+1
                    if last[i]==un_l[0]:
                        c3=c3+1
                    if last[i]==un_l[1]:
                        c4=c4+1
            gi=(1-(c1/n1)**2-(c2/n1)**2)*n1/len(temp)+(1-(c3/n2)**2-(c4/n2)**2)*n2/len(temp)
            m1=m2=0
            for i in range(len(last)):
                if last[i]==un_l[0]:
                    m1=m1+1
                if last[i]==un_l[1]:
                    m2=m2+1
                d=1-(m1/len(data))**2-(m2/len(data))**2
                gini=d-gi
        else:
            gini=0.5
    else:
        gini=0
    return gini

def node(data):
    if len(data)!=0:
        if data.shape[1]>2:
            blank=np.array([])
            for i in range(data.shape[1]-2):
                blank=np.append(blank,gini_gain(data,i))
            max_gain=np.max(blank)
            i=np.argwhere(blank==max_gain)
            return max_gain,int(i[0])
        return 0,0
    else:
        return None

def split(data):
#    b1=b2=np.array([]) #storing the array
    a1=a2=np.array([]) #storing  the arguments
    un=np.unique(data[:,int(node(data)[1])])
    for k in range(len(data[:,-2])):
        if data[k,int(node(data)[1])]==un[0]:
#            b1=np.append(b1,data[k,int(node(data)[1])])
            a1=np.append(a1,k)
        elif data[k,int(node(data)[1])]==un[1]:
#            b2=np.append(b2,data[k,int(node(data)[1])])
            a2=np.append(a2,k)

    return a1,a2 #index

def construct(data):
        l,r=split(data)
        if len(l)>0 and len(r)>0:
            f1=np.zeros((data.shape[1]-1))
            f2=np.zeros((data.shape[1]-1))
            for i in range(len(l)):
                e1=np.array([])
                for j in range(data.shape[1]):
                    if j!=node(data)[1]:                
                        dummy=data[int(l[i]),j]
                        e1=np.append(e1,dummy)
                f1=np.vstack([f1,e1])
            f1=f1[1:,:]
            for i in range(len(r)):
                e2=np.array([])
                for j in range(data.shape[1]):
                    if j!=node(data)[1]:                
                        dummy2=data[int(r[i]),j]
                        e2=np.append(e2,dummy2)
                f2=np.vstack([f2,e2])
            f2=f2[1:,:]            
            return f1,f2
        elif len(l)>0 and len(r)==0:
            f1=np.zeros((data.shape[1]-1))
            for i in range(len(l)):
                e1=np.array([])
                for j in range(data.shape[1]):
                    if j!=node(data)[1]:                
                        dummy=data[int(l[i]),j]
                        e1=np.append(e1,dummy)
                f1=np.vstack([f1,e1])
            f1=f1[1:,:]
            return f1,np.array([])
        elif len(r)>0 and len(l)==0:
            f2=np.zeros((data.shape[1]-1))
            for i in range(len(r)):
                e2=np.array([])
                for j in range(data.shape[1]):
                    if j!=node(data)[1]:                
                        dummy=data[int(r[i]),j]
                        e2=np.append(e2,dummy)
                f2=np.vstack([f1,e2])
            f2=f2[1:,:]
            return np.array([]),f2
        else:
            return np.array([]),np.array([])

def majority_vote(data,split):
    if len(data)!=0:
        if data.shape[1]>2:
            tr1=data[:,split]
            trm=data[:,-2]
            arg=data[:,-1]
            c1=c2=c3=c4=0
            arg_1=np.array([])
            arg_2=np.array([])
            un1=np.unique(tr1)
            unm=np.unique(trm)
            if len(un1)==2 and len(unm)==2:
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
                        elif c1==c2:
                            if unm[0]>unm[1]:
                                for j in range(len(arg_1)):
                                    trm[j]=unm[0]
                            else:
                                for j in range(len(arg_1)):
                                    trm[j]=unm[1]                                                                   
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
                            if unm[0]>unm[1]:
                                for j in range(len(arg_1)):
                                    trm[j]==unm[0]
                            else:
                                for j in range(len(arg_1)):
                                    trm[j]==unm[1] 
                    for i in range(len(trm)):
                        d_t2[int(float(arg[i])),-2]=trm[i]
            else:
                for i in range(len(tr1)):
                    trm[i]=unm[0]
            return None
        elif data.shape[1]==2:
            c1=c2=0
            un=np.unique(data[:,-2])
            if len(un)==1:
                for j in range(len(data)):
                    data[j,0]==un[0]
            else:
                for j in range(len(data)):
                    if data[j,0]==un[0]:
                        c1=c1+1
                    else:
                        c2=c2+1
                if c1>c2:
                    for j in range(len(data)):
                        data[j,0]=un[0]
                elif c1<c2:
                    for j in range(len(data)):
                        data[j,0]=un[1]
                else:
                    if un[0]>un[1]:
                        for j in range(len(data)):
                            data[j,0]=un[0]
                    else:
                        for j in range(len(data)):
                            data[j,0]=un[1]
            for k in range(len(data)):
                d_t2[int(float(data[k,-1])),-2]=data[k,0]
            return None   
        else:
            return None



def pleasework(data,n):
    print(node(data))
    print(data)
    if len(data)!=0:
#        for i in range(data.shape[1]-2):
#            print(gini_gain(data,i))
#        print(split(data))
        if n==nmax:
            if len(data)!=0:
                majority_vote(data,node(data)[1])
                return None
            else:
                return None
        if len(data)!=0:
            if node(data)[0]==0.5:
                majority_vote(data,node(data)[1])
                return None
        #    if len(data)==0:
        #            return None
            if n<nmax:
                n=n+1
                left,right=construct(data)
                pleasework(left,n)
                pleasework(right,n)
                return None
    else:
        return None

pleasework(d_t2,0)

d_train=np.genfromtxt(StringIO(train),delimiter='',skip_header=1,dtype='str')
arg1=np.array([])
for i in range(len(d_train)):
    arg1=np.append(arg1,i)
d_train=np.column_stack([d_train,arg1])


f1=open(f'{sys.argv[4]}',"w")
for i in range(len(d_t2)):
    f1.write(str(d_t2[i,-2])+"\n")
f1.close()


num=0;
for i in range(len(d_t2)):
    if d_train[i,-2]!=d_t2[i,-2]:
        num=num+1
        
e_train=num/len(d_train)

print(f'error(train):{e_train}')

d_t2=d_test

d_test=np.genfromtxt(StringIO(test),delimiter='',skip_header=1,dtype='str')
arg2=np.array([])
for i in range(len(d_test)):
    arg2=np.append(arg2,i)
d_test=np.column_stack([d_test,arg2])

pleasework(d_t2,0)

num=0;
for i in range(len(d_t2)):
    if d_test[i,-2]!=d_t2[i,-2]:
        num=num+1
        
e_test=num/len(d_test)

print(f'error(test): {e_test}')

f2=open(f'{sys.argv[5]}',"w")
for i in range(len(d_test)):
    f2.write(str(d_t2[i,-2])+"\n")
f2.close()

f3=open(f'{sys.argv[6]}',"w")
f3.write(f'error(train): {e_train}\nerror(test): {e_test}')
f3.close()





    