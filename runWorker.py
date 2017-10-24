#############################################################
# This functions implements the ADMM iterations at one worker
# This functions can be run in parallel using the Process Module
#############################################################


# from opf_admm_model import opf_admm_model
from time import time
from numpy import dot
from pypower.idx_cost import COST

def runWorker(ID, s, output):
	print("Worker %d initialized successfully!" % (ID,))
	nu = 0 #iteration count
	itermax = s.admmopt.iterMaxlocal #get maximum iteration
	flag = False
	# itermax = 1
	# while not s.pb['converge']:
	while nu <= itermax and not flag:
		if s.recvmsg: 
			s.update_z()
			s.choose_max_rho()
			
		start_time = time()
		result = s.pipslopf_solver()
		end_time = time()

		if result['eflag'] and nu % 20 == 0:
			print('Subproblem %d at iteration %d solved!' % (ID, nu) )
			# print("Time for solving subproblem %d: %ssecs to %ssecs" % (ID, start_time, end_time))

		s.update_x()

		if s.recvmsg:  # not the initialization
			s.update_y()
			s.update_rho()

		# check convergence
		flag = s.converge()
		s.recvmsg = {}  # clear the received messages 
		s.send()
		s.recv()
		nu += 1

	# record results 
	print("Worker %d finished!" % (ID,))
	print("Local iteration of worker %d is %d" % (ID, nu))
	# calculate local generation cost
	gencost = s.region['gencost']
	pg = s.var['Pg']
	objValue = dot(gencost[:,COST], pg ** 2) + dot(gencost[:,COST + 1], pg) + sum(gencost[:,COST + 2])

	varDual = {'ymd': s.var['ymd'], 'yad': s.var['yad'], 'yms': s.var['yms'],
		'yas': s.var['yas']}
	varPrimal = {'Vm': s.var['Vm'], 'Va': s.var['Va'], 
		'Pg': s.var['Pg'], 'Qg': s.var['Qg']}
	Result = {'objValue': objValue, 'varPrimal': varPrimal, 'varDual': varDual, 'localiter': nu, 
		'primalgap': s.pb['primalgap'], 'dualgap': s.pb['dualgap']}
	output.put((ID - 1, Result))
	