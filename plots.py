import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys


# network bandwdith (Mbit/sec) required in CAKE
# bcast operation requires log(p) stages of data transfer through the 
# broadcast tree of p processors
def cake_network_bw_theoretical(M,N,K,alpha,p,mh,t):
	return (((M*K*math.ceil(N/float(alpha*p*mh)) + K*N*math.ceil(M/float(p*mh))*(math.floor(math.log(p,2))+1) + M*N)*32) / float(t)) / 1e6



def cake_network_bw_observed(M,N,K,alpha,p,mh,t):
	lines = open("mpiP_result/%d-1" % p,'r').read().split("\n")
	start = 0
	for i in lines:
		if 'Aggregate Sent Message Size' in i:
			start = lines.index(i)
			break
	#
	offset = 3
	total_io = 0
	#
	for i in range(20):
		x = lines[start + offset + i].split()
		try:
		    total_io += float(x[3])
		except (ValueError, IndexError) as e:
		    break
	#
	# return ((total_io / t)*8) / 1e6
	io_scatter = M*K*math.ceil(N/float(alpha*p*mh))*4
	return (((total_io + io_scatter) / t)*8) / 1e6



# def plot_network_bw(M,N,K,alpha,p,mh,ntrials,fname = 'cake_c2_network_bw'):
# 	plt.figure(figsize = (6,5))
# 	fig, ax1 = plt.subplots() 
# 	a = range(1,p+1)
# 	# t = [239.916827,120.200325,78.963147,60.911122]
# 	t = [372.046111,195.982044,116.223985,110.035147]
# 	fig, ax1 = plt.subplots() 
# 	color = 'tab:red'
# 	ax1.set_xlabel('number of hosts',fontsize = 16) 
# 	ax1.set_ylabel('Speedup in Computation Time',color = color, fontsize = 16 ) 
# 	ax1.plot(a, [t[0]/t[i] for i in range(len(t))], label = "observed speedup" , color = color) 
# 	ax1.plot(a, range(1,p+1), color = color, label = "ideal speedup", linestyle='dashed') 
# 	ax1.tick_params(axis ='y') 
# 	ax1.set_xlim(1,p)
# 	ax1.set_xticks(range(1,p+1,1))
# 	# Adding Twin Axes to plot using dataset_2
# 	ax2 = ax1.twinx() 
# 	color = 'tab:green'
# 	ax2.set_ylabel('network bandwidth (Mbit/sec)',color = color, fontsize = 16) 
# 	ax2.plot(a, [cake_network_bw_theoretical(M,N,K,alpha,a[i],mh,t[i]) for i in range(len(a))], 
# 		label = "CAKE Theoretical", color = color, linestyle='dashed') 
# 	ax2.plot(a, [cake_network_bw_observed(M,N,K,alpha,a[i],mh,t[i]) for i in range(len(a))], 
# 		label = "CAKE Observed", color = color) 
# 	ax2.tick_params(axis ='y') 
# 	ax2.set_ylim([0, 1200])
# 	plt.title('(a) Speedup and Network BW Usage in CAKE', fontsize = 18)
# 	ax1.legend(loc = "lower right", prop={'size': 12})
# 	ax2.legend(loc = "upper left", prop={'size': 12})
# 	plt.show()
# 	# plt.savefig("%s.pdf" % fname, bbox_inches='tight')
# 	plt.clf()
# 	plt.close('all')



def plot_speedup(M,N,K,alpha,p,mh,ntrials,fname = 'cake_c2_speedup'):
	plt.figure(figsize = (6,5))
	a = range(1,p+1)
	# t = [239.916827,120.200325,78.963147,60.911122]
	t = [372.046111,195.982044,116.223985,110.035147]
	# t = [192.260280,102.113120,62.782158,52.832168]
	color = 'tab:red'
	plt.plot(a, [t[0]/t[i] for i in range(len(t))], label = "observed speedup" , color = color) 
	plt.plot(a, range(1,p+1), color = color, label = "ideal speedup", linestyle='dashed') 
	plt.xlabel('number of hosts',fontsize = 16) 
	plt.ylabel('Speedup in Computation Time', fontsize = 16 ) 
	plt.xticks(range(1,p+1,1))
	plt.title('(a) Speedup in CAKE', fontsize = 18)
	plt.legend(loc = "lower right", prop={'size': 12})
	plt.show()
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.clf()
	plt.close('all')



def plot_network_bw(M,N,K,alpha,p,mh,ntrials,fname = 'cake_c2_network_bw'):
	plt.figure(figsize = (6,5))
	a = range(1,p+1)
	# t = [239.916827,120.200325,78.963147,60.911122]
	t = [372.046111,195.982044,116.223985,110.035147]
	# t = [192.260280,102.113120,62.782158,52.832168]
	plt.ylabel('network bandwidth (Mbit/sec)', fontsize = 16) 
	plt.plot(a, [cake_network_bw_theoretical(M,N,K,alpha,a[i],mh,t[i]) for i in range(len(a))], 
		label = "CAKE Theoretical", linestyle='dashed') 
	plt.plot(a, [cake_network_bw_observed(M,N,K,alpha,a[i],mh,t[i]) for i in range(len(a))], 
		label = "CAKE Observed") 
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.show()
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.clf()
	plt.close('all')




if __name__ == '__main__':
	# plot_speedup(10000,10000,10000,1,4,836,ntrials=int(sys.argv[1]))
	# plot_network_bw(10000,10000,10000,1,4,836,ntrials=int(sys.argv[1]))
	plot_speedup(14000,14000,14000,1,4,1672,ntrials=int(sys.argv[1]))
	plot_network_bw(14000,14000,14000,1,4,1672,ntrials=int(sys.argv[1]))

