###################################################################
# This function gets all tie lines of the partitioned system
# tieline field: Fbus, Tbus, Fbus_region, Tbus_region, tlglobalidx
# Author: Junyao Guo
###################################################################

from numpy import ix_, hstack, transpose, reshape
from numpy import flatnonzero as find
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import BUS_AREA

def findTieline(bus, branch):
	tl = find(bus[branch[:, F_BUS].astype(int), BUS_AREA] != \
		bus[branch[:, T_BUS].astype(int), BUS_AREA])
	ntl = len(tl)
	busBD = branch[ix_(tl, [F_BUS, T_BUS])].astype(int)
	busBDregion = hstack((bus[ix_(busBD[:, 0], [BUS_AREA])], \
		bus[ix_(busBD[:, 1], [BUS_AREA])])).astype(int)
	tieline = hstack((busBD, busBDregion, tl.reshape((ntl,1))))
	return tieline, ntl