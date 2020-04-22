from environment import MountainCar
import sys
import numpy as np

mod=sys.argv[1]
episodes=int(sys.argv[4])
nmax=int(sys.argv[5])
e=float(sys.argv[6])
gamma=float(sys.argv[7])
alpha=float(sys.argv[8])
nactions=3

env=MountainCar(mode=mod)

def dotproduct(d,w):
    prod=0
    for key,val in d.items():
        prod=prod+w[key]*val
    return prod

def creatematrix(d):
    sparse=np.zeros(env.state_space)
    for key,val in d.items():
        sparse[int(key)]=val
    return sparse

def qlearning(episodes,nmax,alpha,gamma):
    rewe=[]
    w=np.zeros((env.state_space,nactions))
    b=0
    for episode in range(episodes):
        r=0
        state_init=env.reset()
        for i in range(nmax):
            qsa=dotproduct(state_init,w)+b
            if np.random.uniform(0,1)<1-e:
                action=np.argmax(qsa)                
            else:
                action=np.random.randint(nactions)                
            next_state,reward,status=env.step(action)
            qsa2=dotproduct(next_state,w)+b
            w2=np.zeros((env.state_space,nactions))
            grad=creatematrix(state_init)
            w2[:,action]=grad
            w=w-alpha*(qsa[action]-(reward+gamma*np.max(qsa2)))*w2
            b=b-alpha*(qsa[action]-(reward+gamma*np.max(qsa2)))
            r=r+reward
            state_init=next_state
            if status:
                print(f'Top Reached at Iteration {i+1}')
                break
            elif i==nmax-1:
                print(f'{nmax} Iterations reached for {episode+1} episode')
        rewe.append(r)  
    return rewe,w,b

rewe,w,b=qlearning(episodes,nmax,alpha,gamma)
x=np.reshape(w,(nactions*env.state_space))
x=np.append(b,x)

f1=open(sys.argv[2],"w")
for line in x:
    f1.write(f'{line}'+'\n')
f1.close()

f2=open(sys.argv[3],"w")
for line in rewe:
    f2.write(f'{line}'+'\n')
f2.close()

# q_learning.py raw weight_out_raw.txt returns_out.txt 4 200 0.05 0.99 0.01 
# q_learning.py tile weight_out_tile.txt returns_out.txt 4 200 0.05 0.99 0.01    
