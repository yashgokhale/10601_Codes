import sys

ab=[]
with open(sys.argv[4], 'r') as f:
    dicti2 = f.readlines()
    for line in dicti2:
        line=line.rstrip()
        line=line.split()
        ab.append(line)
dictionary=dict(ab)

parag1=[]
with open(sys.argv[1],'r') as fd:
    mn=fd.readlines()
    for line in mn:
        key,val=(line.split('\t'))
        parag1.append({key:val})

parag2=[]
with open(sys.argv[2],'r') as fd:
    mn=fd.readlines()
    for line in mn:
        key,val=(line.split('\t'))
        parag2.append({key:val})

parag3=[]
with open(sys.argv[3],'r') as fd:
    mn=fd.readlines()
    for line in mn:
        key,val=(line.split('\t'))
        parag3.append({key:val})    

kk=[]
vv=[]
for key,val in dictionary.items():
    kk.append(key)  
    vv.append(val)
        
def model1(parag,output):
    em=''
    for i in range(len(parag)):
        d=dict()
        tr=parag[i]
        for key,val in tr.items():
            ans=key
            k=val
        em=em+ans+'\t'
        s=k.split()
        for word in s:
            if word in d:
                d[word]=d[word]+1
            else:
                d[word]=1
        aa=0
        for k1,v1 in d.items():
            aa=aa+1
            try:
                a=kk.index(k1)
                if aa==len(d):
                    em=em+str(a)+':1'
                else:
                    em=em+str(a)+':1\t'
            except ValueError:
                n=1
        em=em+'\n'
    f1=open(output,"w")
    f1.write(em)
    f1.close()
    return None

def model2(parag,output):
    em=''
    for i in range(len(parag)):
        d=dict()
        tr=parag[i]
        for key,val in tr.items():
            ans=key
            k=val
        em=em+ans+'\t'
        s=k.split()
        for word in s:
            if word in d:
                d[word]=d[word]+1
            else:
                d[word]=1
        aa=0
        for k1,v1 in d.items():
            aa=aa+1
            if v1<4:
                try:
                    a=kk.index(k1)
                    if aa==len(d):
                        em=em+str(a)+':1'
                    else:                      
                        em=em+str(a)+':1\t'
                except ValueError:
                    n=1
        em=em+'\n'
    f1=open(output,"w")
    f1.write(em)
    f1.close()
    return None

if int(sys.argv[8])==1:
    model1(parag1,sys.argv[5])
    model1(parag2,sys.argv[6])
    model1(parag3,sys.argv[7])
else:
    model2(parag1,sys.argv[5])
    model2(parag2,sys.argv[6])
    model2(parag3,sys.argv[7])
    
#C:\Users\yashg>feature.py train_data.tsv valid_data.tsv test_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1