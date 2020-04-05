import sys
import numpy as np
from io import StringIO

#Sparsing the data into w_train,w_test

with open(sys.argv[1], 'r') as f:
    train = f.read()
d_train=np.genfromtxt(StringIO(train),delimiter='\n',dtype='str')
w_train=[]
ul=[]
for line in d_train:
    f=line.split(' ')
    dp=dict()
    tt=[]
    for k in f:
        ss=k.split('_')
        tt.append(ss)
        dp.update({ss[0]:ss[1]})
    w_train.append(tt)
    ul.append(dp) 
    
#Reference indices
def argu(data):
    xx=np.array([0,0])
    for i in range(len(data)):
        temp=np.array([i,data[i]])
        xx=np.vstack([xx,temp])
    xx=xx[1:,:]
    return xx
with open(sys.argv[2], 'r') as f:
    word= f.read()
iword=np.genfromtxt(StringIO(word),delimiter='\n',dtype='str')
with open(sys.argv[3], 'r') as f:
    tag= f.read()
itag=np.genfromtxt(StringIO(tag),delimiter='\n',dtype='str')
arg_word=argu(iword)
arg_tag=argu(itag)
aa=arg_tag[:,1]
ab=arg_word[:,1]

#initialization
#prior
ff2=np.array([])
for i in w_train:
    aa=i[0][1]
    ff2=np.append(ff2,aa)
prior=np.array([])
for i in range(len(arg_tag)):
    ttt=np.argwhere(arg_tag[i,1]==ff2)
    prior=np.append(prior,len(ttt))
pars_prior=[]
for k in prior:
    s=(k+1)/(np.sum(prior)+len(prior))
    pars_prior.append(s)
e1=''
for i in pars_prior:
    e1=e1+str("%e" %i)+'\n'
f1=open(sys.argv[4],'w')
f1.write(e1)
f1.close()

#transition
u=[]
for i in w_train:
    t=[]
    for j in range(len(i)):
        t.append(i[j][1])
    u.append(t)
xs=0
transition=np.zeros([len(arg_tag),len(arg_tag)])
for i in range(len(arg_tag[:,1])):
    for j in range(len(arg_tag[:,1])):
        c=0
        for m in u:
            for k in range(len(m)-1):
                    if m[k]==arg_tag[i,1] and m[k+1]==arg_tag[j,1]:
                        c=c+1
        transition[i,j]=c
        xs=xs+1
pars_trans=np.zeros([len(arg_tag),len(arg_tag)])
for i,k in enumerate(transition):
    for l,j in enumerate(k):
           pars_trans[i,l]=(j+1)/(np.sum(k)+len(transition))
e2=''
for i in pars_trans:
    for j in i:
        e2=e2+str("%e" %j)+' '
    e2=e2+'\n'
f1=open(sys.argv[5],'w')
f1.write(e2)
f1.close()   

#emission
emission=np.zeros([len(arg_tag),len(arg_word)])
aa=arg_tag[:,1]
ab=arg_word[:,1]
dd2=dict()
with open(sys.argv[1], 'r') as f:
    train = f.read()
d_train=np.genfromtxt(StringIO(train),delimiter='\n',dtype='str')
w_train=[]
count=0
for line in d_train:
    f=line.split(' ')
    tt=[]
    for k in f:
        count=count+1
        ss=k.split('_')
        w_train.append((ss[0],ss[1]))
        dd2.update({count:[ss[0],ss[1]]})
uu=0
for key,val in dd2.items():
    uu=uu+1
    a,b=val
    s1,=np.argwhere(a==ab)
    s2,=np.argwhere(b==aa)
    emission[int(s2[0]),int(s1[0])]=emission[int(s2[0]),int(s1[0])]+1
pars_emm=np.zeros([len(arg_tag),len(arg_word)])
for i,k in enumerate(emission):
    for l,j in enumerate(k):
           pars_emm[i,l]=(j+1)/(np.sum(k)+len(arg_word))

e3=''
for i in pars_emm:
    for j in i:
        e3=e3+str("%e" %j)+' '
    e3=e3+'\n'
f1=open(sys.argv[6],'w')
f1.write(e3)
f1.close()   

#py learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt parsprior.txt parstrans.txt parsemit.txt