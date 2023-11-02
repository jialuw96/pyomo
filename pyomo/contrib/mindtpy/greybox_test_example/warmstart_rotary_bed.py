import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 

# This function takes cov_y (discrete decisions) as input 
# return a model where the continuous variables (FIM) are updated with inputs
def initialize(mod):

    with open("rotary_bed_unit_FIM", 'rb') as f:
        unit_fim_list = pickle.load(f)
        
    new_fim = np.zeros((5,5))
    
    ### compute FIM 
    def eval_fim2(m, a, b):
        """
        Evaluate fim 
        FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses

        a, b: dimensions for FIM, iterate in parameter set 
        """
        if a <= b: 
            summi = 0 
            for i in m.n_responses:
                for j in m.n_responses:
                    if j>=i:
                        summi += m.cov_y[i,j].value*unit_fim_list[i*m.num_measure_dynamic_flatten+j][a][b]
                    else:
                        summi += m.cov_y[j,i].value*unit_fim_list[i*m.num_measure_dynamic_flatten+j][a][b]


            return summi
        else:
            summi = 0 
            for i in m.n_responses:
                for j in m.n_responses:
                    if j>=i:
                        summi += m.cov_y[i,j].value*unit_fim_list[i*m.num_measure_dynamic_flatten+j][b][a]
                    else:
                        summi += m.cov_y[j,i].value*unit_fim_list[i*m.num_measure_dynamic_flatten+j][b][a]


            return summi
            
 
    for a in mod.DimFIM:
        for b in mod.DimFIM:
            if a<=b:
                dynamic_initial_element = eval_fim2(mod, a,b)
                mod.TotalFIM[a,b].value = dynamic_initial_element
                new_fim[a,b] = dynamic_initial_element
                new_fim[b,a] = dynamic_initial_element
                
                ### update greybox
                grey_box_name = "ele_"+str(a)+"_"+str(b)
                mod.my_block.egb.inputs[grey_box_name].value = dynamic_initial_element
                
    
    # initialize determinant 
    _, det = np.linalg.slogdet(new_fim)
    mod.my_block.egb.outputs["log_det"] = det 
          
    # manual and dynamic install initial 
    def total_dynamic_exp(m):
        return sum(m.cov_y[i,i].value for i in range(m.n_static_measurements, m.num_measure_dynamic_flatten))
        
    total_dynamic_initial = total_dynamic_exp(mod)
    mod.total_number_dynamic_measurements.value = total_dynamic_initial 
      
    ### cost constraints
    def cost_compute(m):
        """Compute cost
        cost = static-cost measurement cost + dynamic-cost measurement installation cost + dynamic-cost meausrement timepoint cost 
        """
        return sum(m.cov_y[i,i].value*m.cost_list[i] for i in m.n_responses)+sum(m.if_install_dynamic[j].value*m.dynamic_install_cost[j-m.n_static_measurements] for j in m.DimDynamic)

    cost_init = cost_compute(mod)
    mod.cost.value = cost_init
        
    return mod


update_mod = initialize(mod)