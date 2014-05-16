import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
################################################################################
#################matrix size
N =20
 
################################################################################
#################constants
basal = 0.01
high = 0.99
dx = 1/float(N+3) ####length=N+4 elements ---> o--o--o--o--o (length-1)connections
dy = 1/float(N+3)
#noise=0.0001
#invtemp0=float(1)

maxiter=5
step=0.0000001
lam=1

c=np.zeros((N,N))
################################################################################
################################################################################
################################################################################
####starting conditions####
###########################linear
a0 = np.ones((N,N+2))
b0 = np.ones((N+2,N))
a0 = a0-np.linspace(basal,1,num=N+2)
a0 = a0[:,1:-1]
b0 = b0-np.repeat(np.linspace(basal,1,num=N+2),N).reshape(N+2,-1)
b0 = b0[1:-1,:]
###########################exponential
#Z = np.zeros((N,N))
#e=np.linspace(0,2,num=N)
#for i in range(0,N):
#	e[i]=math.exp(-e[i])
#a0 = Z+e
#b0 = np.repeat(e, N)
#b0=np.reshape(b0, (N,N))
#############################random
#a0=np.random.random_sample((N, N))
#b0=np.random.random_sample((N, N))
#############################zeros or basal
#a0=basal*np.ones((N, N))
#b0=basal*np.ones((N, N)) 
##################################################################################
################################################################################
################################################################################
################################################################################
##################################################################################
####functions####
#################plot####(input two (NxN)-matrices)
def plot(a,b):
	xspace = np.arange(0, 1, 1/float(N))
	yspace = np.arange(0, 1, 1/float(N))
	
	xspacev, yspacev = np.meshgrid(xspace, yspace)
	
	fig = plt.figure()
	ax = fig.add_subplot(211, projection='3d')
	ax.plot_wireframe(xspacev, yspacev, a, rstride=1, cstride=1)

	ax = fig.add_subplot(212, projection='3d')
	ax.plot_wireframe(xspacev, yspacev, b, rstride=1, cstride=1)

	plt.show()
	return None
###################smoothval######(input two (NxN)-matrices,pq)=value of smooth term **
def smoothval(mu,i,j):
	val=(mu[i+1,j]-2*mu[i,j]+mu[i-1,j])/(dx*dx)-(mu[i,j+1]-2*mu[i,j]+mu[i,j-1])/(dy*dy)
	return val

###################fisherval######(input two (NxN)-matrices,pq)=value inside fisherinfo brackets *
def fisherval(a,b,i,j):
	val=(a[i+1,j]-a[i-1,j])*(b[i,j+1]-b[i,j-1])/(4*dx*dy)-(a[i,j+1]-a[i,j+1])*(b[i+1,j]-b[i-1,j])/(4*dx*dy)
	return val

##################fisherinfo######(input two (N+4xN+4)-matrices)=calculates fisherinfo matrix at each n x n element 
def inversefisher(a,b):
	for i in range(0,N):
		for j in range(0,N):
			c[i,j]=pow(fisherval(a,b,i+1,j+1),-2)	
	return c.sum()
#################smootherror######(input two (N+4xN+4)-matrices)=calculates laplace-equ error matrix at each n x n element 
def smootherror(a,b):
	for i in range(0,N):
		for j in range(0,N):
			c[i,j]=lam*(pow(smoothval(a,i+1,j+1),2)+pow(smoothval(b,i+1,j+1),2))
	return c.sum()

###################derivative######(input two (NxN)-matrices,pq,mu)=calculates the derivative wrt to change in a or b of whole expression(inversefisherinfo+smoothing)
def derivative(a,b,p,q,mu):
	if mu==0:
		der=(-pow(fisherval(a,b,p-1,q),-3)*(b[p-1,q+1]-b[p-1,q-1])+pow(fisherval(a,b,p+1,q),-3)*(b[p+1,q+1]-b[p+1,q-1])+pow(fisherval(a,b,p,q-1),-3)*(b[p+1,q-1]-b[p-1,q-1])-pow(fisherval(a,b,p,q+1),-1)*(b[p+1,q+1]-b[p-1,q+1]))/(2*dx*dy)+2*lam*((smoothval(a,p-1,q)+smoothval(a,p+1,q)-2*smoothval(a,p,q))/(dx*dx)+(smoothval(a,p,q-1)+smoothval(a,p,q+1)-2*smoothval(a,p,q))/(dy*dy))

	else:
		der=(-pow(fisherval(a,b,p-1,q),-3)*(a[p-1,q+1]-a[p-1,q-1])+pow(fisherval(a,b,p+1,q),-3)*(a[p+1,q+1]-a[p+1,q-1])+pow(fisherval(a,b,p,q-1),-3)*(a[p+1,q-1]-a[p-1,q-1])-pow(fisherval(a,b,p,q+1),-1)*(a[p+1,q+1]-a[p-1,q+1]))/(2*dx*dy)+2*lam*((smoothval(b,p-1,q)+smoothval(b,p+1,q)-2*smoothval(b,p,q))/(dx*dx)+(smoothval(b,p,q-1)+smoothval(b,p,q+1)-2*smoothval(b,p,q))/(dy*dy))

	return der

###################derivativesmooth######(input two (N+4xN+4)-matrices,pq,mu)=calculates the derivative wrt to change in a or b of whole expression(inversefisherinfo+smoothing)
def derivativesmooth(a,b,p,q,mu):
	if mu==0:
		der=2*lam*((smoothval(a,p-1,q)+smoothval(a,p+1,q)-2*smoothval(a,p,q))/(dx*dx)+(smoothval(a,p,q-1)+smoothval(a,p,q+1)-2*smoothval(a,p,q))/(dy*dy))
	else:
		der=2*lam*((smoothval(b,p-1,q)+smoothval(b,p+1,q)-2*smoothval(b,p,q))/(dx*dx)+(smoothval(b,p,q-1)+smoothval(b,p,q+1)-2*smoothval(b,p,q))/(dy*dy))

	return der
###################addnofluxbound######((NxN)-matrix)
def noflux(mat):
	mattop=mat[0:2,:]
	matbottom=mat[N-2:N,:]
	mat=np.row_stack((mattop,mat))
	mat=np.row_stack((mat,matbottom))
	matleft=mat[:,0:2]
	matright=mat[:,N-2:N]
	mat=np.column_stack((matleft,mat))
	mat=np.column_stack((mat,matright))
	return mat

###################addabsbound######((NxN)-matrix)
def absbound(mat):
	mat1=np.zeros((2,N))
	mat=np.row_stack((mat1,mat))
	mat=np.row_stack((mat,mat1))
	mat2=np.zeros((N+4,2))
	mat=np.column_stack((mat2,mat))
	mat=np.column_stack((mat,mat2))
	return mat
#############################routine##############################################



####initialize
a=np.copy(a0)
b=np.copy(b0)

#print inversefisher(a,b)

dera=np.zeros((N,N))  ##der-matrix wrt change in a (mu=0)
derb=np.zeros((N,N))  ##der-matrix wrt change in b (mu=0) 

print a
plot(a0,b0) 

#####stepping
for t in xrange(0,maxiter):
	vara=noflux(a)
	varb=noflux(b)
	print smootherror(vara,varb)
	#####calc derivative matrix
	for i in range(0,N):
		for j in range(0,N):
			dera[i,j]=derivativesmooth(vara,varb,i+2,j+2,0)
			derb[i,j]=derivativesmooth(vara,varb,i+2,j+2,1)
	###do step

	a=a+step*dera
	b=b+step*derb
	plot(a,b)

##################################################################################
##################################################################################	
##################################################################################	







