import math 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy import optimize
import random
################################################################################
#################matrix size
n = 10
N =n+1 
################################################################################
#################constants
basal = 0.01
high = 0.99
dx = 1/float(N-1)
I=0.680625 
invtemp0=float(10)
maxsteps=100000

c=np.zeros((N,N))
################################################################################
################################################################################
################################################################################
####starting conditions####
###########################linear
#a0 = np.ones((N,N+2))
#b0 = np.ones((N+2,N))
#a0 = a0-np.linspace(basal,1,num=N+2)
#a0 = a0[:,1:-1]
#b0 = b0-np.repeat(np.linspace(basal,1,num=N+2),N).reshape(N+2,-1)
#b0 = b0[1:-1,:]
###########################exponential
Z = np.zeros((N,N))
e=np.linspace(0,2,num=N)
for i in range(0,N):
	e[i]=math.exp(-e[i])
a0 = Z+e
b0 = np.repeat(e, N)
b0=np.reshape(b0, (N,N))
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
##################fisherinfo-fehler######(input two (NxN)-matrices)
def fehler(a,b):
	for i in range(0,N-1):
		for j in range(0,N-1):
			c[i,j]=(a[i,j+1]-a[i,j])*(b[i+1,j]-b[i,j])/(dx*dx)-(a[i+1,j]-a[i,j])*(b[i,j+1]-b[i,j])/(dx*dx)-I
	cval=c[:-1,:-1]
	return np.linalg.norm(cval)
##################fisherinfo-testfehler######(input two (NxN)-matrices,p,q,mu)
def testfehler(a,b,test,p,q,mu):
	if mu==0:
		c[p-1,q]=(test[p-1,q+1]-test[p-1,q])*(b[p,q]-b[p-1,q])/(dx*dx)-(test[p,q]-test[p-1,q])*(b[p-1,q+1]-b[p-1,q])/(dx*dx)-I
		c[p,q]=(test[p,q+1]-test[p,q])*(b[p+1,q]-b[p,q])/(dx*dx)-(test[p+1,q]-test[p,q])*(b[p,q+1]-b[p,q])/(dx*dx)-I
		c[p,q-1]=(test[p,q]-test[p,q-1])*(b[p+1,q-1]-b[p,q-1])/(dx*dx)-(test[p+1,q-1]-test[p,q-1])*(b[p,q]-b[p,q-1])/(dx*dx)-I
	elif mu==1: 
		c[p-1,q]=(a[p-1,q+1]-a[p-1,q])*(test[p,q]-test[p-1,q])/(dx*dx)-(a[p,q]-a[p-1,q])*(test[p-1,q+1]-test[p-1,q])/(dx*dx)-I
		c[p,q]=(a[p,q+1]-a[p,q])*(test[p+1,q]-test[p,q])/(dx*dx)-(a[p+1,q]-a[p,q])*(test[p,q+1]-test[p,q])/(dx*dx)-I
		c[p,q-1]=(a[p,q]-a[p,q-1])*(test[p+1,q-1]-test[p,q-1])/(dx*dx)-(a[p+1,q-1]-a[p,q-1])*(test[p,q]-test[p,q-1])/(dx*dx)-I
	else: print 'problem testfehler'
	cval=c[:-1,:-1]
	return np.linalg.norm(cval)
#####################reheating - increase temperature if stuck ######(input lastgoodone)
def heating(lastgoodone):
	if lastgoodone<100:
		invtemp=invtemp0 
	elif lastgoodone==100:		
		print 'temp rises'
		invtemp=invtemp0/10
	elif lastgoodone<1000:
		invtemp=invtemp0/10
	elif lastgoodone==1000:
		print 'temp still rises'
		invtemp=invtemp0/100
	else:
		invtemp=invtemp0/100
	print "T: {0:f}".format(invtemp)
	return invtemp
#####################ramped stepsize######(input number of step - does something like that make sense?!?!?!)
def step(i): 
	if i<10000000:
		stepsize=0.01
	#elif i<1000000:
	#	stepsize=0.001
	#elif i<1000000000:
	#	stepsize=0.0001
	else: 
		stepsize=0.00001
	return stepsize
		
##################################################################################	
##################################################################################	
#################################initialize#######################################

###counting####
nostep=0
badstep=0
goodstep=0
###stuff###
lastgoodone=0
memory=0
test=np.zeros((N,N))
a=a0
b=b0
error=fehler(a,b)


##################################################################################	
#####################################routine######################################	
##################################################################################	
for i in xrange(0,maxsteps):
	print error
	#if memory==0:
	####draw random element
	p=np.random.choice(n)
	q=np.random.choice(n)
	####decide which morphogen mu and which direction di
	mu=np.random.choice(2)
	di=np.random.choice(2)
	###make small step
	if mu==0:
		test=np.copy(a)
		if di==0:
			test[p,q]=test[p,q]+step(i)
	
		else:
			test[p,q]=test[p,q]-step(i)
	elif mu==1: 
		test=np.copy(b)
		if di==0:
			test[p,q]=test[p,q]+step(i)
		else:
			test[p,q]=test[p,q]-step(i)
	else: print 'problem #mu'

	####look if got better or worse
	newerror=testfehler(a,b,test,p,q,mu)
	delta=error-newerror
	if delta>0:  			##got better
		lastgoodone=0
		memory=1
		error=newerror
		print 'good step + continue in this direction'
		goodstep=goodstep+1	
		if mu==0:		##do step
			a=test
		elif mu==1:
			b=test
		else: print 'weird #mu'
	else: 				##got worse
		lastgoodone=lastgoodone+1
		memory=0
		barriere=math.exp(delta*heating(lastgoodone))
		print "pa: {0:f}".format(barriere)
		u=np.random.random_sample()
		if u<barriere: 		##still allow sometimes
			badstep=badstep+1
			print 'bad step in wrong direction'
			if mu==0:	##do step
				a=test
			elif mu==1:
				b=test
			else: print 'weird #mu'
		else: 
			print 'no step'
			nostep=nostep+1






##################################################################################	
#####################################results######################################	
##################################################################################	

print goodstep
print badstep
print nostep
#plot(a0,b0)
#plot(a,b)









