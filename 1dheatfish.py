import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
from scipy import optimize
################################################################################
#################input
N = 50
dx=1/float(N-1)
lam=0    ####ratio of fisherinfo to heat - contribution
################################################################################
################################################################################
################################################################################
################starting condition  
#                                          constant --- step at 0 from 1 to 0.01
#f0=np.ones((N))*0.01

#                                 			       linear decreasing 
#f0=np.linspace(1,0,num=N)
#f0=np.delete(f0, 0)
#f0=np.delete(f0,-1)
#                                                       exponentially decreasing 
#f0=np.linspace(0,2,num=N)
#for i in range(0,N):
#	f0[i]=math.exp(-f0[i])*500
#f0=np.delete(f0, 0)
#f0=np.delete(f0,-1)
#									  random
f0=np.random.random_sample((N-2))

################################################################################
#############################calc D'
d0=np.zeros((N))
d0[0]=1
d0[-1]=1
dmat0=np.diag(d0)
d1=np.ones((N-1))/(2*dx)
d2=np.ones((N-1))/(2*dx)
d1[-1]=0
d2[0]=0
dmat1=np.diag(d1,k=-1)
dmat2=np.diag(d2,k=1)
d=dmat0+dmat1-dmat2
################################################################################
#############################calc D''
s0=np.ones((N))*-2/(dx*dx)
s0[0]=1
s0[-1]=1
smat0=np.diag(s0)
s1=np.ones((N-1))/(dx*dx)
s2=np.ones((N-1))/(dx*dx)
s1[-1]=0
s2[0]=0
smat1=np.diag(s1,k=-1)
smat2=np.diag(s2,k=1)
s=smat0+smat1+smat2
################################################################################
#################################plot
def plot(mu):
	cells = np.arange(N)/float(N)
	fig=plt.figure()
	rect=fig.patch
	rect.set_facecolor('grey')
	
	ax1=fig.add_subplot(1,1,1, axisbg='white')
	ax1.plot(cells,mu,'c', linewidth=3.3)
	ax1.set_xlabel('space')
	ax1.set_ylabel('morphogene concentration')

	plt.show()
	return None
################################################################################
####score function ---- fin is vector of size N-2, which is actually optimized (boundary elements excluded)
val=np.ones((N))
valfish=np.ones((N))
valheat=np.ones((N))
def score(fin): ####both parts contribute: fish*lam+heat 
	f=np.append([1],fin)
	f=np.append(f,[0.01])
	valfish=((np.dot(d,f)*np.dot(d,f))-(np.dot(s,f)*f))*((np.dot(d,f)*np.dot(d,f))-(np.dot(s,f)*f))
	valheat=np.dot(s,f)*np.dot(s,f)
	val=lam*valfish+valheat
	print np.sum(val)
	return np.sum(val)
def heat(fin): #####just heat contributes
	f=np.append([1],fin)
	f=np.append(f,[0.01])
	valheat=np.dot(s,f)*np.dot(s,f)
	print np.sum(valheat)
	return np.sum(valheat)
def fish(fin):####just fish contributes
	f=np.append([1],fin)
	f=np.append(f,[0.01])
	valfish=((np.dot(d,f)*np.dot(d,f))-(np.dot(s,f)*f))*((np.dot(d,f)*np.dot(d,f))-(np.dot(s,f)*f))
	print np.sum(valfish)
	return np.sum(valfish)
################################################################################
################################gradient
st=np.ones((N))
def grad(fin):
	f=np.append([1],fin)
	f=np.append(f,[0.01])
	st=s.transpose()
	der=np.dot(st,s)
	der=np.dot(der,f)
	return 2*der
################################################################################


################################################################################
###############################routine


sol = optimize.fmin_cg(score,f0)

realstart=np.append([1],f0)
realstart=np.append(realstart,[0.01])

realsol=np.append([1],sol)
realsol=np.append(realsol,[0.01])
print realstart
print realsol

plot(realstart)
plot(realsol)




