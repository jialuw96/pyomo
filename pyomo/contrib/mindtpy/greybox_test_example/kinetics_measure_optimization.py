import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 


# ======= change here =======
# aim to solve budgets from 1000 to 5000
budget_opt = 4500

# choose what solutions to initialize with 
initial_option = "milp_A"
#"milp_A": mixed-integer problem solutions, trace as objective function 
#"nlp_D": relaxed NLP problem solutions of the current problem

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


num_static = len(static_ind)
num_dynamic  = len(dynamic_ind)
num_total = num_static + num_dynamic*Nt



# ==== initialization strategy ==== 
if initial_option == "milp_A":
    curr_results = np.linspace(1000, 5000, 11)
    file_name_pre, file_name_end = './kinetics_results_reduced/milp_', '_a'

elif initial_option == "nlp_D":
    curr_results = np.linspace(1000, 5000, 41)
    file_name_pre, file_name_end = './kinetics_results_reduced/nlp_', '_d'
    
    
curr_results = set([int(curr_results[i]) for i in range(len(curr_results))])

if budget_opt in curr_results: # use an existed initial solutioon
    curr_budget = budget_opt

else:
    # find the closest budget
    curr_min_diff = float("inf")
    curr_budget = 5000

    for i in curr_results:
        if abs(i-budget_opt) < curr_min_diff:
            curr_min_diff = abs(i-budget_opt)
            curr_budget = i

    print("using solution at", curr_budget, " too initialize")


y_init_file = file_name_pre+str(curr_budget)+file_name_end
fim_init_file = file_name_pre+'fim_'+str(curr_budget)+file_name_end

    
# initialize solution 
with open(y_init_file, 'rb') as f:
    init_cov_y = pickle.load(f)

# round the fractional NLP solution. round down, so that cost constraint is not violated
for i in range(11):
    for j in range(1):
        if init_cov_y[i][j] > 0.99:
            init_cov_y[i][j] = int(1)
        else:
            init_cov_y[i][j] = int(0)
            

total_manual_init = 0 
dynamic_install_init = [0]

for i in range(3,11):
    if init_cov_y[i][i] > 0.01:
        total_manual_init += 1 
        
        i_pos = int((i-3)/8)
        dynamic_install_init[i_pos] = 1
    
# initialize FIM
with open(fim_init_file, 'rb') as f:
    fim_prior = pickle.load(f)
    

    
mip_option = True
objective = ObjectiveLib.D
sparse_opt = True
fix_opt = False

manual_num = 10

num_dynamic_time = np.linspace(0,60,9)

static_dynamic = [[1,3]]
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
                    total_manual_num_init=total_manual_init)

mod = calculator.solve(mod, mip_option=mip_option, objective = objective)

fim_result = np.zeros((4,4))
for i in range(4):
    for j in range(i,4):
        fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])
        
print(fim_result)  
print('trace:', np.trace(fim_result))
print('det:', np.linalg.det(fim_result))
print(np.linalg.eigvals(fim_result))

print("Pyomo OF:", pyo.value(mod.Obj))
print("Log_det:", np.log(np.linalg.det(fim_result)))

ans_y, sol_y = calculator.extract_solutions(mod)
print('pyomo calculated cost:', pyo.value(mod.cost))
print("if install dynamic measurements:")
print(pyo.value(mod.if_install_dynamic[3]))
#print(pyo.value(mod.if_install_dynamic[4]))
#print(pyo.value(mod.if_install_dynamic[5]))