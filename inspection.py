import sys

from io import StringIO

import numpy as np

#------taking in user inputs
with open(sys.argv[1], 'r') as f:
    train = f.read()

#----converting user data to array
d_train=np.genfromtxt(StringIO(train),delimiter='',skip_header=1,dtype='str')

un=np.unique(d_train[:,-1])

count=np.zeros(len(un))

for i in range(len(d_train[:,-1])):
    for j in range(len(un)):
        if d_train[i,-1]==un[j]:
            count[j]=count[j]+1

#---error

err=1-(np.max(count))/len(d_train)

#---gini impurity
gin=0

for j in range(len(count)):
    gin=gin+(count[j]/len(d_train))*(1-(count[j]/len(d_train)))
    
file=open(f'{sys.argv[2]}',"w")
file.write(f'gini_impurity: {gin}\nerror: {err}')
file.close()
    
    
            