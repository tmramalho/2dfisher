import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
from scipy import optimize
################################################################################
#################input
N = 10 ###total size of variable grid without boundaries
M = 13
dx=1/float(N-1)
dy=1/float(M-1)
T = M*N
#####just for routine
s=0.00001
maxiter=10000000
#starting condition - random
f0=np.random.random_sample((T))

###############################################################################
###############################################################################
#####################definitions
###############################################################################
def plot(f):
    xspace = np.linspace(0,1,num=M+2)
    yspace = np.linspace(0,1,num=N+2)
    xspacev, yspacev = np.meshgrid(xspace, yspace)
    f=np.reshape(f, (N+2, M+2))
    fig = plt.figure()
    a = fig.add_subplot(111, projection='3d')
    a.plot_wireframe(xspacev, yspacev, f, rstride=1, cstride=1)
 
    #ax = fig.add_subplot(212, projection='3d')
    #ax.plot_wireframe(xspacev, yspacev, b, rstride=1, cstride=1)
 
    plt.show()
    return None
###############################################################################
####make (N+2)*(M+2)-vector with boundary conditions from N*M-vector
def larger(f):
	f=np.reshape(f, (N, M))
	bo=f[1,:]      ####no flux on top and bottom
	bu=f[-2,:]
	ma=np.row_stack((bo,f))
	ma=np.row_stack((ma,bu))
	l=np.ones((N+2)) ##### left boundary ->1
	r=np.ones((N+2))*0.01   #### right boundary ->0.01
	ma=np.column_stack((l,ma))
	ma=np.column_stack((ma,r))
	return ma.flatten()
###############################################################################
####make N*M-vector out of (N+2)*(M+2)-vector, which should really be optimized 
def smaller(ftot):
	ftot=np.reshape(ftot, (N+2, M+2))
	f=ftot[1:-1,1:-1]
	return f.flatten()
###############################################################################
###############################################################################
####built matrix mat so that fnew=fold+mat*fold*s
d0=np.zeros(((M+2)*(N+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d0[(i-1)*(M+2)+j-1]=-2/(dx*dx)-2/(dy*dy) 
mat0=np.diag(d0)
##
d1=np.zeros(((M+2)*(N+2)-(M+2)))
for i in range(1,N+1):
	for j in range(2,M+2):
		d1[(i-1)*(M+2)+j-1]=1/(dx*dx)
mat1=np.diag(d1,k=-(M+2))
##
d2=np.zeros(((M+2)*(N+2)-1))
for i in range(2,N+2):
	for j in range(2,M+2):
		d2[(i-1)*(M+2)+j-1-1]=1/(dy*dy)
mat2=np.diag(d2,k=-1)
##
d3=np.zeros(((M+2)*(N+2)-1))
for i in range(2,N+2):
	for j in range(2,M+2):
		d3[(i-1)*(M+2)+j-1]=1/(dy*dy)
mat3=np.diag(d3,k=1)
##
d4=np.zeros(((M+2)*(N+2)-(M+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d4[(i-1)*(M+2)+j-1]=1/(dx*dx)
mat4=np.diag(d4,k=(M+2))
mat=mat0+mat1+mat2+mat3+mat4
###############################################################################
###############################################################################
########initialize and save starting vector
print f0
#f0tot=larger(f0)
#ftot=f0tot






################optimization
def score(fin):
	f=larger(fin)
	val=np.dot(mat,f)*np.dot(mat,f)
	return np.sum(val)

#plot(f0tot)
#plot(ftot)



sol = optimize.fmin_cg(score,f0)


print sol
plot(larger(sol))

















################routine for validation of mat
#for n in xrange(0,maxiter):
#	ftot=ftot+s*np.dot(mat,ftot) 
	######ensure no flux condition
#	for j in range(2,M+2):
#		ftot[j-1]=ftot[j-1+M+2]
#		ftot[(N+1)*(M+2)+j-1]=ftot[N*(M+2)+j-1]





