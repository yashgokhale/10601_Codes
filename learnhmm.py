import numpy as np
import sys

d_train=np.genfromtxt(sys.argv[1],delimiter='\n',dtype='str')
w_train=[]
for i in d_train:
    ww=i.split(' ')
    w_train.append(ww)

iword=np.genfromtxt(sys.argv[2],delimiter='\n',dtype='str')
dword=dict()
for i in range(len(iword)):
    dword.update({iword[i]:i})
itag=np.genfromtxt(sys.argv[3],delimiter='\n',dtype='str')
dtag=dict()
for i in range(len(itag)):
    dtag.update({itag[i]:i})

#prior
prior=np.zeros(len(itag))
for w in w_train:
    for j in range(len(w)):
        if j==0:
            a1,b1=w[j].split('_')
            iw=dword[a1]
            it=dtag[b1]
            prior[it]=prior[it]+1   
pars_prior=np.zeros(len(itag))
for i in range(len(prior)):
    pars_prior[i]=(prior[i]+1)/(np.sum(prior)+len(prior))
e1=''
for i in pars_prior:
    e1=e1+str("%.18e" %i)+'\n'
f1=open(sys.argv[4],'w')
f1.write(e1)
f1.close()

#trans
transition=np.zeros([len(itag),len(itag)])
for w in w_train:
    for j in range(len(w)):
        if j<len(w)-1:   
            a1,b1=w[j].split('_')
            iw=dword[a1]
            it=dtag[b1]
            a2,b2=w[j+1].split("_")
            iw2=dword[a2]
            it2=dtag[b2]
            transition[it,it2]=transition[it,it2]+1
pars_trans=np.zeros([len(itag),len(itag)])
for i,k in enumerate(transition):
    for l,j in enumerate(k):
           pars_trans[i,l]=(j+1)/(np.sum(k)+len(transition))  
e2=''
for i in pars_trans:
    for j in i:
        e2=e2+str("%.18e" %j)+' '
    e2=e2+'\n'
f1=open(sys.argv[6],'w')
f1.write(e2)
f1.close()   
    
#emission       
emission=np.zeros([len(itag),len(iword)])
for w in w_train:
    for j in range(len(w)):          
        a2,b2=w[j].split("_")
        iw2=dword[a2]
        it2=dtag[b2]
        emission[it2,iw2]=emission[it2,iw2]+1
pars_emm=np.zeros([len(itag),len(iword)])
for i,k in enumerate(emission):
    for l,j in enumerate(k):
           pars_emm[i,l]=(j+1)/(np.sum(k)+emission.shape[1])
e3=''
for i in pars_emm:
    for j in i:
        e3=e3+str("%.18e" %j)+' '
    e3=e3+'\n'
f1=open(sys.argv[5],'w')
f1.write(e3)
f1.close()

#py learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt parsprior.txt parsemit.txt parstrans.txt
#py learnhmm.py toytrain.txt toy_index_to_word.txt toy_index_to_tag.txt parsprior.txt parsemit.txt parstrans.txt