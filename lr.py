import numpy as np
import sys

vote1=[]
args1=[]

ab=[]
with open(sys.argv[4], 'r') as f:
    dicti2 = f.readlines()
    for line in dicti2:
        line=line.rstrip()
        line=line.split()
        ab.append(line)
w=[0]*(len(ab)+1)

dddd=0

with open(sys.argv[1],'r') as fd:
    mn=fd.readlines()
    for row in mn:
        a,b=(row.split('\t',maxsplit=1))
        vote1.append(a)
        aa=((b.split("\t")))
        ll=[elem.strip().split(':') for elem in aa]
        bbb=dict(ll)
        args1.append(bbb)

def sigmoid(X,W):
    product=0
    for key,val in X.items():
        product=product+float(W[int(key)+1])*float(val)
    return 1/(1+np.exp(-product)) 

def sgd2(X,y,rate,n_epochs):
    coef=[0.0 for i in range(len(w))] #1 additional for intercept
    l=[]
    for e in range(n_epochs):
        for i in range(len(X)):
            pr=0
            row=X[i]
            a={'-1':'1'}
            a.update(row)
            yp=sigmoid(a,coef)
            for key,val in a.items():
                coef[int(key)+1]=coef[int(key)+1]+rate*float(val)*(float(y[i])-yp)
    print('l',l)
    return coef

pars=sgd2(args1,vote1,0.1,int(sys.argv[8]))

less=0
more=0
def sign(X,w):
    global less,more
    product=0
    a={'-1':'1'}
    a.update(X)
    for key,val in a.items():
        product=product+w[int(key)+1]*float(val)
    dd=1/(1+np.exp(-product))
    if dd<=0.5 and dd>=0:
        less=less+1
        return 0
    elif dd>0.5 and dd<=1:
        more=more+1
        return 1
    if dd==0.5:
        print('bingo')
    else:
        print('Something is wrong',dd)
        return dd  
    
def comparison(arg,vote,name):
    global pars
    c=0
    l=''
    for i in range(len(arg)):
        l=l+str(sign(arg[i],pars))+'\n'
        if (sign(arg[i],pars))!=int(vote[i]):
            c=c+1
    f1=open(name,"w")
    f1.write(l)
    f1.close()
    return c

a=comparison(args1,vote1,sys.argv[5])
a1=a/len(args1)

args2=[]
vote2=[]
dddd=0
with open(sys.argv[3],'r') as fd:
    mn=fd.readlines()
    for row in mn:
        a,b=(row.split('\t',maxsplit=1))
        vote2.append(a)
        aa=((b.split("\t")))
        ll=[elem.strip().split(':') for elem in aa]
        bbb=dict(ll)
        args2.append(bbb)

b=comparison(args2,vote2,sys.argv[6])
b1=b/len(args2)

print(a1,b1)

f2=open(sys.argv[7],"w")
f2.write(f'error(train): {a1}\nerror(test): {b1}')
f2.close()

    #lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60