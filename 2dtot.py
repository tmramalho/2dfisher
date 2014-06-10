import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
from scipy import optimize
################################################################################
#################input
N = 3 ###total size of variable grid without boundaries
M = 5
dx=1/float(N-1)
dy=1/float(M-1)
T = M*N
lam=100
#####just for routine
s=0.00001
maxiter=10000000
#starting condition - random
f0a=np.random.random_sample((T))
f0b=np.random.random_sample((T))
###############################################################################
###############################################################################
#####################definitions
###############################################################################
####make (N+2)*(M+2)-vector with boundary conditions from N*M-vector!!!FOR A!!!
def largera(f):
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
####make (N+2)*(M+2)-vector with boundary conditions from N*M-vector!!!FOR B!!!
def largerb(f):
	f=np.reshape(f, (N, M))
	l=f[:,1]     ####no flux on left and right
	r=f[:,-2]
	ma=np.column_stack((l,f))
	ma=np.column_stack((ma,r))
	bo=np.ones((M+2))  #####top boundary ->1
	bu=np.ones((M+2))*0.01   #### bottom boundary ->0.01
	ma=np.row_stack((bo,ma))
	ma=np.row_stack((ma,bu))
	return ma.flatten()
###############################################################################
####make N*M-vector out of (N+2)*(M+2)-vector, which should really be optimized 
def smaller(ftot):
	ftot=np.reshape(ftot, (N+2, M+2))
	f=ftot[1:-1,1:-1]
	return f.flatten()
###############################################################################
#####################plot
def plot(f):
    xspace = np.linspace(0,1,num=M+2)
    yspace = np.linspace(0,1,num=N+2)
    xspacev, yspacev = np.meshgrid(xspace, yspace)
    fina=np.split(f, 2)[0]
    finb=np.split(f, 2)[1]
    fina=np.reshape(largera(fina), (N+2, M+2))
    finb=np.reshape(largerb(finb), (N+2, M+2))
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_wireframe(xspacev, yspacev, fina, rstride=1, cstride=1)
 
    ax = fig.add_subplot(212, projection='3d')
    ax.plot_wireframe(xspacev, yspacev, finb, rstride=1, cstride=1)

    plt.show()
    return None
###############################################################################
####built matrices 
d0=np.zeros(((M+2)*(N+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d0[(i-1)*(M+2)+j-1]=1
mat0=np.diag(d0)
##
d1=np.zeros(((M+2)*(N+2)-(M+2)))
for i in range(1,N+1):
	for j in range(2,M+2):
		d1[(i-1)*(M+2)+j-1]=1
mat1=np.diag(d1,k=-(M+2))


##
d2=np.zeros(((M+2)*(N+2)-1))
for i in range(2,N+2):
	for j in range(2,M+2):
		d2[(i-1)*(M+2)+j-1-1]=1
mat2=np.diag(d2,k=-1)
##
d3=np.zeros(((M+2)*(N+2)-1))
for i in range(2,N+2):
	for j in range(2,M+2):
		d3[(i-1)*(M+2)+j-1]=1
mat3=np.diag(d3,k=1)
##
d4=np.zeros(((M+2)*(N+2)-(M+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d4[(i-1)*(M+2)+j-1]=1
mat4=np.diag(d4,k=(M+2))
##### mat=2nd deriv(x) + 2nd deriv(y)
mat=mat0*(-2/float(dx*dx)-2/float(dy*dy))+mat1/float(dx*dx)+mat2/float(dy*dy)+mat3/float(dy*dy)+mat4/float(dx*dx)
##### matx=first deriv(x)
matx=mat4/float(dx*dx)-mat1/float(dx*dx)
##### maty=first deriv(y)
maty=mat3/float(dy*dy)-mat2/float(dy*dy)

################score functions
def scoreheat(f):
	fina=np.split(f, 2)[0]
	finb=np.split(f, 2)[1]
	fa=largera(fina)
	vala=np.dot(mat,fa)*np.dot(mat,fa)
	fb=largerb(finb)
	valb=np.dot(mat,fb)*np.dot(mat,fb)
	return np.sum(vala)+np.sum(valb)

def scorefish(f):
	fina=np.split(f, 2)[0]
	finb=np.split(f, 2)[1]
	fa=largera(fina)
	fb=largerb(finb)
	val=(np.dot(matx,fa)*np.dot(maty,fb)-np.dot(maty,fa)*np.dot(matx,fb))*(np.dot(matx,fa)*np.dot(maty,fb)-np.dot(maty,fa)*np.dot(matx,fb))
	realval=pow(smaller(val),-2)
	return np.sum(realval)
def score(f):
	return scorefish(f)+scoreheat(f)*lam

###############################################################################
###############################################################################
f=np.append(f0a,f0b)

for i in range(0,10):
	sol = optimize.fmin_cg(score,f)
	plot(f)
	plot(sol)
	lam=lam/float(10)
	f=sol

################routine for validation of mat
#for n in xrange(0,maxiter):
#	ftot=ftot+s*np.dot(mat,ftot) 
	######ensure no flux condition
#	for j in range(2,M+2):
#		ftot[j-1]=ftot[j-1+M+2]
#		ftot[(N+1)*(M+2)+j-1]=ftot[N*(M+2)+j-1]





