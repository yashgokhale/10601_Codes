import numpy as np
import sys

#Data Reading
iword=np.genfromtxt(sys.argv[2],delimiter='\n',dtype='str')
itag=np.genfromtxt(sys.argv[3],delimiter='\n',dtype='str')
p=np.genfromtxt(sys.argv[4])
b=np.genfromtxt(sys.argv[5])
print(b)
a=np.genfromtxt(sys.argv[6])
print(a)
d_test=np.genfromtxt(sys.argv[1],delimiter='\n',dtype='str')
w_test=[]
attr=[]
words=[]
if d_test.size==1:
    d_test=str(d_test)
    xx=d_test.split(' ')
    for k in xx:
        ss=k.split('_')
        a1,b1=ss
        words.append(a1)
        attr.append(b1)
    words=[words]
    attr=[attr]
else:
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
    return warg,l
    
def write(word):
    a0,l=forwardbackward(word)
    em=''
    for i in range(len(word)):
        if i<len(word)-1:
            em=em+word[i]+'_'+itag[int(a0[i])]+' '  
        elif i==len(word)-1:
            em=em+word[i]+'_'+itag[int(a0[i])]
    return em,l,a0

output=''
likelihood=0
count=0
total=0

for i in range(len(words)):
    print('Predicting word number:',i)
    w1,l1,a0=write(words[i])
    likelihood=likelihood+l1
    output=output+w1+'\n'
#    a0,a1=forwardbackward(words[i])
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

#forwardbackward.py testwords.txt index_to_word.txt index_to_tag.txt parsprior.txt parsemit.txt parstrans.txt predicted.txt metrics.txt
#forwardbackward.py toytest.txt toy_index_to_word.txt toy_index_to_tag.txt parsprior.txt parsemit.txt parstrans.txt predicted.txt metrics.txt