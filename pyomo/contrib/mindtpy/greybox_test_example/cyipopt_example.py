import numpy as np
import pandas as pd
import pyomo.environ as pyo
from greybox_generalize import LogDetModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock

def build_model_external(m):
    ex_model = LogDetModel(n_parameters=5)
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)

def create_model():
    m = pyo.ConcreteModel()

    n_param = 5 
    m.DimFIM = pyo.Set(initialize=range(n_param))

    def DimFIMhalf_init(m):
        return ((a,b) for a in m.DimFIM for b in range(a, n_param))

    m.DimFIM_half = pyo.Set(dimen=2, initialize=DimFIMhalf_init)

    def identity(m,a,b):
        return 1 if a==b else 0

    # discrete decisions
    m.TotalFIM = pyo.Var(m.DimFIM_half, initialize=identity, bounds=(-5, 5), within=pyo.Binary)

    # continuous decisions
    #m.TotalFIM = pyo.Var(m.DimFIM_half, initialize=identity, bounds=(-5, 5), within=pyo.Reals)
    
    # fix decisions 
    #m.TotalFIM.fix()
    
    def _model_i(b):
        build_model_external(b)
    m.my_block = pyo.Block(rule=_model_i)
    
    for i in range(n_param):
        for j in range(i, n_param):
            def eq_fim(m):
                return m.TotalFIM[i,j] == m.my_block.egb.inputs["ele_"+str(i)+"_"+str(j)]
            
            con_name = "con"+str(i)+str(j)
            m.add_component(con_name, pyo.Constraint(expr=eq_fim))
    
    # add objective
    m.Obj = pyo.Objective(expr=m.my_block.egb.outputs['log_det'], sense=pyo.maximize)

    return m 

mod = create_model()

solver = pyo.SolverFactory('cyipopt')
solver.config.options['hessian_approximation'] = 'limited-memory' 
additional_options={'max_iter':3000, 'output_file': 'console_output',
                    'linear_solver':'mumps'}

for k,v in additional_options.items():
    solver.config.options[k] = v
solver.solve(mod, tee=True)