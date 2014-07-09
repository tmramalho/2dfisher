import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
from scipy import optimize
################################################################################
#################input
N = 10 ###total size of variable grid without boundaries
M = 10
dx=1/float(N-1)
dy=1/float(M-1)
T = M*N
lam=100
#####just for routine
#s=0.000001
#maxiter=20000
###########################starting condition - random
#f0a=np.random.random_sample((T))
#f0b=np.random.random_sample((T))
###########################exponential     -  only SQUARE lattice!!!!!!!!!!!!!!
Z = np.zeros((N,N))
e=np.linspace(0,2,num=N)
for i in range(0,N):
	e[i]=math.exp(-e[i])
a0 = Z+e
b0 = np.repeat(e, N)
b0=np.reshape(b0, (N,N))
f0a=a0.flatten()
f0b=b0.flatten()
###########################linear
#a0 = np.ones((N,N+2))
#b0 = np.ones((N+2,N))
#a0 = a0-np.linspace(0.01,1,num=N+2)
#a0 = a0[:,1:-1]
#b0 = b0-np.repeat(np.linspace(0.01,1,num=N+2),N).reshape(N+2,-1)
#b0 = b0[1:-1,:]
#f0a=a0.flatten()
#f0b=b0.flatten()
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
d00=np.ones(((M+2)*(N+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d00[(i-1)*(M+2)+j-1]=0
mat00=np.diag(d00)

##
d0=np.ones(((M+2)*(N+2)))
for i in range(2,N+2):
	for j in range(2,M+2):
		d0[(i-1)*(M+2)+j-1]=(-2/float(dx*dx)-2/float(dy*dy))
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
mat=mat0+mat1/float(dx*dx)+mat2/float(dy*dy)+mat3/float(dy*dy)+mat4/float(dx*dx)
##### matx=first deriv(x)
matx=mat00+mat4/float(dx*dx)-mat1/float(dx*dx)
##### maty=first deriv(y)
maty=mat00+mat3/float(dy*dy)-mat2/float(dy*dy)

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
	val=(np.dot(matx,fa)*np.dot(maty,fb)-np.dot(maty,fa)*np.dot(matx,fb))
	realval=fina*fina*finb*finb*pow(smaller(val),-2)
	mean=np.mean(realval) 			###calc mean fisher info
	return np.sum(pow(realval-mean,2))	###substract mean fisher info from real value to get how much it fails
def score(f):
	print 'fish {}.'.format(scorefish(f))	###to see how they contribute
	print 'heat {}.'.format(scoreheat(f)*lam)	
	return scorefish(f)+scoreheat(f)*lam
def testscore(f):
	print 'heat {}.'.format(scoreheat(f))
	return scoreheat(f)
###############################################################################
###############################################################################
###############optimization
f=np.append(f0a,f0b)


######optimize just heat equation
#sol = optimize.fmin_cg(testscore,f)
#plot(f)
#plot(sol)

######optimize both with decreasing lambda
for i in range(10):
	sol = optimize.fmin_cg(score,f)
	print i
	plot(f)
	plot(sol)
	lam=lam/float(10)
	f=sol






###############################################################################
###############################################################################
################routine for validation of mat

#ftota=largera(f0a)
#ftotb=largerb(f0b)
#fges=np.append(f0a,f0b)
#plot(fges)
#print ftota.shape
#print mat.shape
################
#for n in xrange(0,maxiter):
	#ftota=ftota+s*np.dot(mat,ftota) 
	#ftotb=ftotb+s*np.dot(mat,ftotb)
	######ensure no flux condition above and below
	#for j in range(2,M+2):####above and below
	#	ftota[j-1]=ftota[j-1+M+2]
	#	ftota[(N+1)*(M+2)+j-1]=ftota[N*(M+2)+j-1]
	#for i in range(0,N+2):####left and right
	#	ftotb[i*(M+2)]=ftotb[i*(M+2)+2]
	#	ftotb[(i+1)*(M+2)-1]=ftotb[(i+1)*(M+2)-2]
#fa=smaller(ftota)
#fb=smaller(ftotb)
#fgesnew=np.append(fa,fb)
#plot(fgesnew)



