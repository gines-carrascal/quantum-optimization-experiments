#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
# Import Grover’s algorithm and components classes
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.operators import PauliExpectation, CVaRExpectation
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import LinearEqualityToPenalty
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit import execute, Aer
from qiskit.aqua import aqua_globals
from qiskit.optimization.converters import InequalityToEquality
from qiskit.optimization.converters import IntegerToBinary

import numpy as np
import matplotlib.pyplot as plt
import time
from docplex.mp.model import Model

aqua_globals.random_seed = 123456
aqua_globals.massive=True

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# setup aqua logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log




# Import the QuantumInstance module that will allow us to run the algorithm on a simulator and a quantum computer
from qiskit.aqua import QuantumInstance

# Grover's dictionary is used to wrap all the necessary parameters in one dictionary. 
# The following is the dictionary we will use for Grover's Search.
"""
optim_dict = {
  "docplex_mod": 'mdl',
  "quantum_instance": Backend,
  "shots": 1024,
  "print":boolean,
  "solver":'method',
  "optimizer":'SPSA',
  "maxiter":'100',
  "depth":'1',
  "alpha":0.35,
}
"""

# Define our Optimisation function. This is the function that will be called by the quantum-aggregator class.
def optimize_portfolio(dictionary):
    #dictionary["expression"]
    result ={}
    # case to 
    qp = QuadraticProgram()
    qp.from_docplex(dictionary['docplex_mod'])

    if dictionary['print']:
        print('### Original problem:')
        print(qp.export_as_lp_string())
    

    #classical solution
    # solve classically as reference
    if dictionary['solver'] == 'classic':
        t_00 = time.perf_counter()
        opt_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
        t_0 = time.perf_counter() - t_00
        result['computational_time'] = t_0
        result['result'] = opt_result
        #print('Time:',t_0)
        #print(opt_result)
    elif dictionary['solver'] == 'vqe':
        
        conv = InequalityToEquality()
        qp1 = conv.convert(qp)
        if dictionary['print']:
            print('### InequalityToEquality:')
            print(qp1.export_as_lp_string())
        
        #second coverter to be avoided
        conv2 = IntegerToBinary()
        qp2 = conv2.convert(qp1)
        if dictionary['print']:
            print('### IntegerToBinary:')
            print(qp2.export_as_lp_string())
    
        #Converter: Convert the problem to an unconstrained problem, with a penalty
        conv3 = LinearEqualityToPenalty(penalty=dictionary['penalty'])
        qp3 = conv3.convert(qp2)
        if dictionary['print']:
            print('### LinearEqualityToPenalty:')
            print(qp3.export_as_lp_string())

        #quantum preparation
        # set classical optimizer
    
        optimizer = dictionary["optimizer"](maxiter=int(dictionary["maxiter"]))

        # set variational ansatz
        var_form = RealAmplitudes(qp.get_num_binary_vars(), reps=int(dictionary["depth"]))
        m = var_form.num_parameters

        # set backend
        backend = Aer.get_backend(dictionary["quantum_instance"])
    
        # initialize CVaR_alpha objective
        cvar_exp = CVaRExpectation(float(dictionary["alpha"]), PauliExpectation())
        cvar_exp.compute_variance = lambda x: [0]  # to be fixed in PR #1373

        # initialize VQE using CVaR
        vqe = VQE(expectation=cvar_exp, optimizer=optimizer, var_form=var_form, quantum_instance=backend)

        # initialize optimization algorithm based on CVaR-VQE
        opt_alg = MinimumEigenOptimizer(vqe)

        # solve problem
        t_00 = time.perf_counter()
        results = opt_alg.solve(qp3)
        t_0 = time.perf_counter() - t_00
        result['computational_time'] = t_0
        result['result'] = results

        # print results
    if dictionary['print']:
        print('### Results:')
        print(result)
        
    return result