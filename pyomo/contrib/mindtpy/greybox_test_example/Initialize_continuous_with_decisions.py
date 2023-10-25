import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 



Nt = 8

max_manual_num = 10
min_interval_num = 10


static_ind = [0,1,2]
dynamic_ind = [3]
all_ind = static_ind+dynamic_ind

num_total = len(all_ind)

all_names_strategy3 = ["CA.static", "CB.static", "CC.static", "CB.dynamic"]


static_cost = [2000, 2000, 2000, 200]
dynamic_cost = [0, 0, 0]
dynamic_cost.extend([400]*len(dynamic_ind))

max_manual = [max_manual_num]*num_total
min_time_interval = [min_interval_num]*num_total

error_cov = [[1, 0.1, 0.1, 1], 
            [0.1, 4, 0.5, 0.1], 
            [0.1, 0.5, 8, 0.1], 
            [1, 0.1, 0.1, 1]]

measure_info = pd.DataFrame({
    "name": all_names_strategy3,
    "Q_index": all_ind,
        "static_cost": static_cost,
    "dynamic_cost": dynamic_cost,
    "min_time_interval": min_time_interval, 
    "max_manual_number": max_manual
})

dataObject = DataProcess()
dataObject.read_jacobian('Q_drop0.csv')
#Q = dataObject.get_Q_list([0,1,2], [0,1,2], Nt)
Q = dataObject.get_Q_list([0,1,2], [1], Nt)


calculator = MeasurementOptimizer(Q, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)


fim_expect = calculator.fim_computation()



# This function takes cov_y (discrete decisions) as input 
# return a model where the continuous variables (FIM) are updated with inputs
def initialize(mod):
    
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
                        summi += m.cov_y[i,j]*calculator.fim_collection[i*m.num_measure_dynamic_flatten+j][a][b]
                    else:
                        summi += m.cov_y[j,i]*calculator.fim_collection[i*m.num_measure_dynamic_flatten+j][a][b]


            return summi
        else:
            summi = 0 
            for i in m.n_responses:
                for j in m.n_responses:
                    if j>=i:
                        summi += m.cov_y[i,j]*calculator.fim_collection[i*m.num_measure_dynamic_flatten+j][b][a]
                    else:
                        summi += m.cov_y[j,i]*calculator.fim_collection[i*m.num_measure_dynamic_flatten+j][b][a]


            return summi
            
    mod.dynamic_initial_FIM = pyo.Expression(mod.DimFIM_half, rule=eval_fim2)
        
    for a in mod.DimFIM:
        for b in mod.DimFIM:
            if a<b:
                mod.TotalFIM[a,b] = mod.dynamic_initial_FIM[a,b]
                
    return mod


update_mod = initialize(mod)