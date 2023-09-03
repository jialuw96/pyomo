#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    minimize,
    SolverFactory,
    value,
    Block
)
from pyomo.common.collections import ComponentMap

from greybox_cyipopt_eg2 import LogModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock


# In[2]:


def build_model_external(m):
    ex_model = LogModel(initial={"X1": 0, "X2":0, "Y1": 0, "Y2": 1, "Y3": 1})
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model)


# In[3]:


# reference: https://github.com/Pyomo/pyomo/blob/main/pyomo/contrib/mindtpy/tests/MINLP_simple.py

def create_model(grey_box = False, continuous_y = False):
    """
    grey_box: if True, use grey-box module to compute OF. If not, use anaytical equation for OF.
    """
    m = ConcreteModel()
    """Set declarations"""
    I = m.I = RangeSet(1, 2, doc='continuous variables')
    J = m.J = RangeSet(1, 3, doc='discrete variables')

    # initial point information for discrete variables
    initY = {
        'sub1': {1: 1, 2: 1, 3: 1},
        'sub2': {1: 0, 2: 1, 3: 1},
        'sub3': {1: 1, 2: 0, 3: 1},
        'sub4': {1: 1, 2: 1, 3: 0},
        'sub5': {1: 0, 2: 0, 3: 0},
    }
    # initial point information for continuous variables
    initX = {1: 0, 2: 0}

    """Variable declarations"""
    # Y can be DISCRETE VARIABLES or CONTINUOUS VARIABLES
    if continuous_y:
        Y = m.Y = Var(J, domain=NonNegativeReals, initialize=initY["sub2"])
    else:
        Y = m.Y = Var(J, domain=Binary, initialize=initY['sub2'])

    # X are CONTINUOUS VARIABLES
    X = m.X = Var(I, domain=NonNegativeReals, initialize=initX)

    """Bound definitions"""
    # x (continuous) upper bounds
    x_ubs = {1: 4, 2: 4}
    for i, x_ub in x_ubs.items():
        X[i].setub(x_ub)

    """Constraint definitions"""
    # CONSTRAINTS
    m.const1 = Constraint(expr=(m.X[1] - 2) ** 2 - m.X[2] <= 0)
    m.const2 = Constraint(expr=m.X[1] - 2 * m.Y[1] >= 0)
    m.const3 = Constraint(expr=m.X[1] - m.X[2] - 4 * (1 - m.Y[2]) <= 0)
    m.const4 = Constraint(expr=m.X[1] - (1 - m.Y[1]) >= 0)
    m.const5 = Constraint(expr=m.X[2] - m.Y[2] >= 0)
    m.const6 = Constraint(expr=m.X[1] + m.X[2] >= 3 * m.Y[3])
    m.const7 = Constraint(expr=m.Y[1] + m.Y[2] + m.Y[3] >= 1)

    """Cost (objective) function definition"""

    if not grey_box:
        m.objective = Objective(
            expr=Y[1] + 1.5 * Y[2] + 0.5 * Y[3] + X[1] ** 2 + X[2] ** 2, sense=minimize
        )
    
    else: 
        def _model_i(b):
            build_model_external(b)
        m.my_block = Block(rule=_model_i)
        
        for i in m.I:
            def eq_inputX(m):
                return m.X[i] == m.my_block.egb.inputs["X"+str(i)]
            
            con_name = "con_X_"+str(i)
            m.add_component(con_name, Constraint(expr=eq_inputX))

        for j in m.J:
            def eq_inputY(m):
                return m.Y[j] == m.my_block.egb.inputs["Y"+str(i)]
            
            con_name = "con_Y_"+str(j)
            m.add_component(con_name, Constraint(expr=eq_inputY))
        
        # add objective
        m.Obj = Objective(expr=m.my_block.egb.outputs['z'], sense=minimize)
    

    return m


# In[4]:

mod = create_model(grey_box=True, continuous_y=True)

# In[ ]:

### If grey_box=False 
#solver = SolverFactory("gurobi", solver_io="python")
#solver.solve(mod, tee=True)

### If grey_box=True
solver = SolverFactory('cyipopt')
solver.config.options['hessian_approximation'] = 'limited-memory' 
additional_options={'max_iter':3000, 'output_file': 'console_output',
                    'linear_solver':'mumps'}

for k,v in additional_options.items():
    solver.config.options[k] = v

solver.solve(mod, tee=True)
# In[ ]:


print(value(mod.X[1]), value(mod.X[2]))
print(value(mod.Y[1]), value(mod.Y[2]), value(mod.Y[3]))


# In[ ]:




