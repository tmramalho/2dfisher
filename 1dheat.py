import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
################################################################################
#################input
N =10
maxiter=10
size=0.001
################################################################################
################################################################################
d=pow((1/float(N-1)),-2)##inverse of dx^2
################################################################################
################################################################################
################starting condition  
#                              step after first element of f from T=500 to T=100
#f0=np.ones((N))*100 
#f0[0]=500
#f0[-1]=100
#                                  linear - nothing should happen to this one... 
f0=np.linspace(500,50,num=N)
#                                                                    exponential 
#f0=np.linspace(0,2,num=N)
#for i in range(0,N):
#	f0[i]=math.exp(-f0[i])*500
################################################################################
################################################################################
################################################################################
#############################calc (D^T*D)-operator 
m0=np.ones((N))*-2*d
m0[0]=1
m0[-1]=1
m1u=np.ones((N-1))*d
m1u[0]=0
m1d=np.ones((N-1))*d
m1d[-1]=0
mat0=np.diag(m0)
mat1u=np.diag(m1u,k=1)
mat1d=np.diag(m1d,k=-1)
mat=mat0+mat1u+mat1d #### this is the D matrix for fixed ends
matt=np.zeros((N,N))
matt=mat.transpose()

der=np.dot(matt,mat)

################################################################################
##############################initialize
grad=np.zeros((N))
f=np.zeros((N))
i=0
f=f0
################################################################################
###############################routine
for i in xrange(0,maxiter):
	print f
	grad=np.dot(der,f)
	f=f+2*grad*size
	

