# OPF-ADMM
Using Alternating Direction Method of Multipliers to solve AC Optimal Power Flow Problem.

Main file ayncADMM_acopf.py implements the application of using ADMM to solve AC OPF problem.

opf_admm_model.py builds the subproblem object.

runWorker implements the ADMM iterations.

Package Pypower 5.1.3 is used in this application.

Message passing among workers are implemented using the Process module.

Detailed algorithm reference: J. Guo, G. Hug, and O. K. Tonguz, “A case for nonconvex distributed optimization in large-scale power systems,” IEEE Transactions on Power Systems, vol. 32, no. 5, pp. 3842–3851, 2017.
