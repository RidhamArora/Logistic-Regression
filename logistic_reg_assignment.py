import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('logisticX.csv')
dataset2 = pd.read_csv('logisticY.csv')

x =  dataset.iloc[:,[0,1]].values
X1 = dataset.iloc[:,0].values
X2 = dataset.iloc[:,1].values
Y =  dataset2.iloc[:,0].values

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)"""


for i in range(99):
    if Y[i]==1:
        plt.scatter(x[i][0],x[i][1],marker='*',color='Blue')
    else:
        plt.scatter(x[i][0],x[i][1],marker='^',color='Orange')
plt.show()

hyper_list = []
def hypothesis(x,w,b):
    h=x[0]*w[0]+x[1]*w[1]+b
    return sigmoid(h)

def sigmoid(z):
    global hyper_list
    zz = 1.0/(1.0+np.exp(-1.0*z))
    hyper_list.append((z,zz))
    return zz

def error(Y,x,w,b):
    
    m=x.shape[0]
    err=0.0
    
    for i in range(m):
        hx=hypothesis(x[i],w,b)
        err += Y[i]*np.log2(hx) + (1-Y[i])*np.log2(1-hx)
        
    print(-err/m)    
    return -err/m

def get_grads(Y,x,w,b):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    
    m = x.shape[0]

    for i in range(m):
        hx = hypothesis(x[i],w,b)
        
        grad_w += (Y[i]-hx)*x[i]
        grad_b += (Y[i]-hx)
        
    grad_w /= m
    grad_b /= m
    
    return [grad_w,grad_b]

def batch_gradient(Y,x,w,b,batch_size=1):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    m=x.shape[0]
    indices=np.arange(m)
    np.random.shuffle(indices)
    indices=indices[:batch_size]
    for i in indices:
        hx = hypothesis(x[i],w,b)
        
        grad_w += (Y[i]-hx)*x[i]
        grad_b += (Y[i]-hx)
        
    grad_w /= m
    grad_b /= m
    
    return [grad_w,grad_b]

def grad_descent(x,Y,w,b,learning_rate=0.1):
    
    err = error(Y,x,w,b)
    [grad_w,grad_b] = batch_gradient(Y,x,w,b,batch_size=50)
    
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    
    return err,w,b

def logistic(x,Y):
    loss = []
    acc = []
    
    w=np.zeros((x.shape[1],))
    b=np.random.random()
    print(w)
    print(b)
    for i in range(200):
        l,w,b = grad_descent(x,Y,w,b,learning_rate = 0.1)
        print(l,w,b)
        loss.append(l)
    
    return loss,acc,w,b

final_loss,final_acc,final_w,final_b = logistic(x,Y)
#final_w = [ 1.56974928,-1.79457347] --- From sklearn
#final_b = [ 0.22567697]  --- From Sklearn
c=np.linspace(-2,10,10)
a=-(final_w[0]*c+final_b)/final_w[1]
for i in range(99):
    if Y[i]==1:
        plt.scatter(x[i][0],x[i][1],marker='*',color='Blue')
    else:
        plt.scatter(x[i][0],x[i][1],marker='^',color='Orange')
cnt=0
for i in range(99):
    if (hypothesis(x[i],final_w,final_b)>0.5 and Y[i]==1) or (hypothesis(x[i],final_w,final_b)<=0.5 and Y[i]==0):
        cnt=cnt+1
acc = (cnt*1.0)/(x.shape[0]*1.0)
plt.plot(c,a,color='black')
plt.show()
plt.plot(final_loss)
plt.show()
print("Accuracy is %f percent"%(acc*100))