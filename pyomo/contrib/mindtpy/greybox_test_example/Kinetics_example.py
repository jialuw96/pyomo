#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib


### Data Process. Set up problem, do not need to change. 
# In[2]:
Nt = 8
max_manual_num = 10
min_interval_num = 10
static_ind = [0,1,2]
dynamic_ind = [3,4,5]
all_ind = static_ind+dynamic_ind
num_total = len(all_ind)
all_names_strategy3 = ["CA.static", "CB.static", "CC.static", 
                      "CA.dynamic", "CB.dynamic", "CC.dynamic"]
static_cost = [2000, # CA
    2000, # CB
     2000, # CC
    200, # CA
    200, # CB
     200] # CC

dynamic_cost = [0]*len(static_ind)
dynamic_cost.extend([400]*len(dynamic_ind))

max_manual = [max_manual_num]*num_total
min_time_interval = [min_interval_num]*num_total

error_cov = [[1, 0.1, 0.1, 1, 0.1, 0.1],
[0.1, 4, 0.5, 0.1, 4, 0.5],
[0.1, 0.5, 8, 0.1, 0.5, 8], 
[1, 0.1, 0.1, 1, 0.1, 0.1], 
[0.1, 4, 0.5, 0.1, 4, 0.5], 
[0.1, 0.5, 8, 0.1, 0.5, 8]]

# variance 
var_list = [1,4,8,1,4,8]
std_list = [np.sqrt(var_list[i]) for i in range(6)]
print(std_list)

corr_original = [[0]*6 for i in range(6)]

for i in range(6):
    for j in range(6):
        corr_original[i][j] = error_cov[i][j]/std_list[i]/std_list[j]

corr_num = 0.5
for i in range(3):
    for j in range(3,6):
        corr_original[i][j] *= corr_num
        corr_original[j][i] *= corr_num 

for i in range(6):
    for j in range(6):
        error_cov[i][j] = corr_original[i][j]*std_list[i]*std_list[j]

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
Q = dataObject.get_Q_list([0,1,2], [0,1,2], Nt)

calculator = MeasurementOptimizer(Q, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)
fim_expect = calculator.fim_computation()


### Solve

num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
num_total = num_static + num_dynamic*Nt

init_cov_y = np.ones((27, 27))


fim_prior = np.asarray([[  4.2421363 ,   2.0974022  , -7.15679971 , -8.94930454],
 [  2.0974022 ,   5.18300713 , -2.8999988 , -21.02439003],
 [ -7.15679971 , -2.8999988  , 12.34045591 , 12.5645852 ],
 [ -8.94930454 ,-21.02439003 , 12.5645852  , 86.54108698]])

### False means relaxed problem, True means mixed integer problem 
mip_option = False

# All other options do not need to be changed
objective = ObjectiveLib.D

# if False. compute OF with equality constraints instead of grey-box module
grey_box_option = False

sparse_opt = True
fix_opt = False

manual_num = 10
budget_opt = 1000

total_manual_init = 0
dynamic_install_init = [0,0,0] 

num_dynamic_time = np.linspace(0,60,9)

static_dynamic = [[0,3],[1,4],[2,5]]
time_interval_for_all = True

dynamic_time_dict = {}
for i, tim in enumerate(num_dynamic_time[1:]):
    dynamic_time_dict[i] = np.round(tim, decimals=2)


mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                      obj=objective, 
                    fix=fix_opt, 
                    upper_diagonal_only=sparse_opt, 
                    num_dynamic_t_name = num_dynamic_time, 
                    manual_number = manual_num, 
                    budget=budget_opt,
                    init_cov_y= init_cov_y,
                    initial_fim = fim_prior,
                    dynamic_install_initial = dynamic_install_init, 
                    static_dynamic_pair=static_dynamic,
                    time_interval_all_dynamic = time_interval_for_all,
                    total_manual_num_init=total_manual_init, 
                                        grey_box = grey_box_option)

mod = calculator.solve(mod, mip_option=mip_option, objective = objective)


fim_result = np.zeros((4,4))
for i in range(4):
    for j in range(i,4):
        fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])

print("Pyomo OF:", pyo.value(mod.Obj))
print("Log_det:", np.log(np.linalg.det(fim_result)))

print("Solutions:") # following are the solutions. Should be integer for MINLP, and float for relaxed problem
ans_y, sol_y = calculator.extract_solutions(mod)
print("if install dynamic measurements:")
print(pyo.value(mod.if_install_dynamic[3]))
print(pyo.value(mod.if_install_dynamic[4]))
print(pyo.value(mod.if_install_dynamic[5]))


# In[ ]:




