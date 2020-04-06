import numpy as np
import sys
from io import StringIO

#Data Reading
with open(sys.argv[2], 'r') as f:
    word= f.read()
iword=np.genfromtxt(StringIO(word),delimiter='\n',dtype='str')
with open(sys.argv[3], 'r') as f:
    tag= f.read()
itag=np.genfromtxt(StringIO(tag),delimiter='\n',dtype='str')

with open(sys.argv[4], 'r') as f:
    prior = f.readlines()
p=np.genfromtxt(prior,delimiter='\n',dtype='float')

b=np.zeros(len(iword))
with open(sys.argv[5], 'r') as f:
    emit= f.readlines()
    for line in emit:
        line=line.rstrip()
        x=line.split(' ')
        x=np.array(x,dtype='float')
        b=np.vstack([b,x])  
b=b[1:,:]

a=np.zeros(len(itag))
with open(sys.argv[6], 'r') as f:
    transition= f.readlines()
    for line in transition:
        line=line.rstrip()
        x=line.split(' ')
        x=np.array(x,dtype='float')
        a=np.vstack([a,x])
a=a[1:,:]

with open(sys.argv[1], 'r') as f:
    test = f.read()
d_test=np.genfromtxt(StringIO(test),delimiter='\n',dtype='str')
w_test=[]
attr=[]
words=[]
for line in d_test:
    f=line.split(' ')
    dp=dict()
    tt=[]
    w1=[]
    w2=[]
    for k in f:
        ss=k.split('_')
        a1,b1=ss
#        print(a1,b1)
        tt.append(ss)
        w1.append(a1)
        w2.append(b1)
    w_test.append(tt)
    attr.append(w2)
    words.append(w1)
    
#Model Training
def forward(sample):
    alpha=np.zeros([len(sample),len(itag)])
    argu=np.array([])
    for i in range(len(alpha)):
        x=np.argwhere(iword==sample[i])
        argu=np.append(argu,x)
    for t in range(len(alpha)):
        if t==0:
            ao=p*b[:,int(argu[t])]
            alpha[t]=ao
        else:
            for j in range(len(itag)):
                prod=0
                for k in range(len(itag)):
                    prod=prod+a[k,j]*alpha[t-1,k]
                alpha[t,j]=b[j,(int(argu[t]))]*prod
    lhood=np.log(np.sum(alpha[-1]))
    return alpha,lhood

def backward(sample):
    beta=np.zeros([len(sample),len(itag)])
    argu=np.array([])
    for i in range(len(beta)):
        x=np.argwhere(iword==sample[i])
        argu=np.append(argu,x)
    l=len(beta)
    for t in range(len(beta)):
        if t==0:
            bo=np.ones(beta.shape[1])
            beta[l-t-1,:]=bo
        else:
            for j in range(len(itag)):
                prod=0
                for k in range(len(itag)):
                    prod=prod+b[k,int(argu[l-t])]*beta[l-t,k]*a[j,k]
                beta[l-t-1,j]=prod
    return beta

def forwardbackward(sample):
    alpha,l=forward(sample)
    beta=backward(sample)
    prob=alpha*beta
    warg=np.zeros(len(sample))
    for i in range(len(prob)):
        warg[i]=np.argmax(prob[i])
    return warg
    
def write(word):
    a0=forwardbackward(word)
    em=''
    for i in range(len(word)):
        em=em+word[i]+'_'+itag[int(a0[i])]+' '  
    return em

output=''
likelihood=0
count=0
total=0
for i in range(len(words)):
    print('Predicting word number:',i)
    likelihood=likelihood+forward(words[i])[1]
    output=output+write(words[i])+'\n'
    a0=forwardbackward(words[i])
    for j in range(len(attr[i])):
        total=total+1
        if itag[int(a0[j])]!=attr[i][j]:
            count=count+1

accuracy=1-count/total

e2=f'Average Log-Likelihood: {likelihood/len(words)}\nAccuracy: {accuracy}'

f1=open(sys.argv[7],'w')
f1.write(output)
f1.close() 

f1=open(sys.argv[8],'w')
f1.write(e2)
f1.close() 

#forwardbackward.py testwords.txt index_to_word.txt index_to_tag.txt hmmprior.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt