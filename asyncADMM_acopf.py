# synchronous ADMM with no message passing
# sequentially solves subproblems and estimate the total time
# Author: Junyao Guo
# May 17, 2017

import time
import os
import numpy as np 
from numpy import flatnonzero as find
import matplotlib.pyplot as plt
# plt.ion()
from scipy.io import loadmat

from os.path import join

from multiprocessing import Process, Pipe, Queue
import queue

from pypower.loadcase import loadcase
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus

from pypower.idx_bus import BUS_TYPE, BUS_AREA, PD, QD
from pypower.idx_gen import GEN_STATUS, GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.idx_cost import COST

from findTieline import findTieline
from runWorker import runWorker
from opf_admm_model import opf_admm_model

os.system("taskset -p 0xff %d" % os.getpid())

#---------------- basic system configuration  ---------------------------------
caseFile = join('/Users/junyaoguo/anaconda/lib/python3.5/site-packages/pypower', 'case14')
ppc = loadcase(caseFile)
##convert to internal numbering, remove out-of-service stuff 
ppc = ext2int(ppc)
baseMVA, bus, gen, branch, gencost = \
		ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
slack = find(bus[:, BUS_TYPE] == 3) # an array of indices
gen_active = find(gen[:, GEN_STATUS] == 1)
genBus = gen[gen_active, GEN_BUS]

## convert to p.u.
bus[:,[PD, QD]] /= baseMVA
gen[:,[PMAX, PMIN, QMAX, QMIN]] /= baseMVA
gencost[:, COST] *= baseMVA ** 2
gencost[:, COST + 1] *= baseMVA

## problem dimensions
nb = bus.shape[0]          # number of buses
nl = branch.shape[0]       # number of branches
ng = len(gen_active)   # number of piece-wise linear costs

## build admittance matrices
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

##---------- partition the system ---------------------------------------------
na = 3 # number of regions
partitionName = 'OP14_2region.mat' 
# partitionName = 'OP118_w05_40region_sizediff.mat'
# partitionName = 'OP118_rd_8region_scale100000.mat'
partitionDir = '/Users/junyaoguo/Dropbox/ADMM/Partition'
partitionFile = join(partitionDir,partitionName)
partition = loadmat(partitionFile)
# partitionnumber = 0
op = partition['OP14'][0]
op[10:] = 3
# op[:] = 1
# op = partition['C'][partitionnumber]
bus[:, BUS_AREA] = op
tieline, ntl = findTieline(bus, branch)

##---------- create all the communication pipes --------------------------------
edges = np.vstack({tuple(sorted(row)) for row in tieline[:, 2:4]}) if tieline.any() else np.array([])
pipes = {}
for edge in edges.tolist():
	fend, tend =  Pipe()
	if edge[0] not in pipes:
		pipes[edge[0]] = {}
	pipes[edge[0]][edge[1]] = fend
	if edge[1] not in pipes:
		pipes[edge[1]] = {}
	pipes[edge[1]][edge[0]] = tend



##----subproblem configuration including local opf and communication pipes-----
problem = []
# result = []
output = Queue()
for i in range(na):
	s = opf_admm_model()
	s.config(i + 1, op, bus, gen, gencost, Ybus, genBus, tieline, pipes, na)
	s.var_init()
	problem.append(s)
	# result.append(s.pipslopf_solver())

##----- run each worker in parallel ---------
procs = []
for i in range(na):
	procs += [Process(target = runWorker, args = (i + 1, problem[i], output))]

start_time = time.time()
start_clock = time.clock()
for proc in procs:
	proc.start()


# TIMEOUT = 70
# bool_list = [True] * na
# start = time.time()
# while time.time() - start <= TIMEOUT:
# 	for i in range(na):
# 		bool_list[i] = procs[i].is_alive()

# 	#print(bool_list)

# 	if np.any(bool_list):  
# 		time.sleep(.1)  
# 	else:
# 		break
# else:
# 	print("Timed out, killing all processes")
# 	for p in procs:
# 		p.terminate()

liveprocs = list(procs)	
results = []
while liveprocs:
	try:
		while 1:
			results.append(output.get(False))
	except queue.Empty:
		pass

	time.sleep(0.5)
	if not output.empty():
		continue

	liveprocs = [p for p in liveprocs if p.is_alive()]

for proc in procs:
	proc.join()

# results = []
# for proc in procs:
# 	while proc.is_alive():
# 		proc.join(timeout = 30)
# 		while True:
# 			try: 
# 				results.append(output.get())
# 			except Empty:
# 				break

## ------------get results ---------------------------
ttime = time.time()
tclock = time.clock()
totaltime = ttime - start_time
clocktime = tclock - start_clock

# results = [output.get() for proc in procs]
results = sorted(results, key=lambda x: x[0])

objTotal = 0
objCent = 8088
# objCent = 129660
for k in range(na):
	if k != results[k][0]:
		print('Error: Result of worker %d not returned!' % (k+1,))
		break
	objTotal += results[k][1]['objValue']

gap = (objTotal - objCent) / objCent * 100
print('The convergence time is %f' % (totaltime,))
print('The convergence clock time is %f' % (clocktime,))
print('The objective function value is %f' % (objTotal,))
print('The gap in objective function is %f %%' % (gap,))

## ------------ plots of convergence -----------------
# fig = plt.figure()
# for k in range(na):
# 	if k != results[k][0]:
# 		print('Error: Result of worker %d not returned!' % (k+1,))
# 		break
# 	pgap = results[k][1]['primalgap']
# 	dgap = results[k][1]['dualgap']
# 	curfig = fig.add_subplot(4, 5, k + 1)
# 	curfig.plot(pgap, color = 'red', linewidth = 2.5, label = 'primal residual')
# 	curfig.plot(dgap, color = 'blue', linewidth = 2.5, label = 'dual residual')
# 	curfig.set_yscale('log')
# 	curfig.legend(loc='upper right')
# plt.show()  


# plt.draw()


# f, df = s1.admmopf_costfcn(result['x'])
# h, g, dh, dg = s1.admmopf_consfcn(result['x'])
# H = s1.admmopf_hessfcn(result['x'])

# s2 = opf_admm_model()
# s2.config(2, op, bus, gen, gencost, Ybus, genBus, tieline, pipes)
# s3 = opf_admm_model()
# s3.config(3, op, bus, gen, gencost, Ybus, genBus, tieline, pipes)

