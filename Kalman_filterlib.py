import numpy as np
import math 

class KalmanFilter:
    def __init__(self,dt,P_k,Qt,Rt,Xk,I):
        self.dt=dt
        self.P_k=P_k
        self.Qt=Qt
        self.Rt=Rt
        self.X_k=Xk
        self.Identity=I
        pass
    def predict(self,U_k,C,A,fd=np.array([])):
        if fd.size==0:
            self.X_k=np.dot(A,U_k)
            self.P_k=np.dot(np.dot(A,self.P_k),np.transpose(A))+self.Qt
        else:
            self.X_k=self.X_k+self.dt*np.dot(fd,U_k)
            self.P_k=self.P_k+self.dt*(np.dot(A,self.P_k)+np.dot(self.P_k,np.transpose(A))+self.Qt)           
            #self.P_k=np.dot(A,self.P_k)+np.dot(self.P_k,np.transpose(A))+self.Qt
    
    def update(self,Zk,C,Hd=np.array([])):
       
        self.Sk=np.linalg.multi_dot([C,self.P_k,np.transpose(C)])+self.Rt 
        self.Sk=np.linalg.inv(self.Sk)
       
            
        Kk=np.linalg.multi_dot([self.P_k,np.transpose(C),self.Sk])
        if Hd.size==0 :
            
            self.X_k=np.dot(Kk,Zk-np.dot(C,self.X_k))+self.X_k
        else:
            self.X_k=np.dot(Kk,Zk-Hd)+self.X_k

        IDOT=(self.Identity-np.dot(Kk,C))
        self.P_k=np.dot(IDOT,self.P_k)