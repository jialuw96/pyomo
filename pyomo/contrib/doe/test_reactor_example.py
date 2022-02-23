#################################################################################################################
# Copyright (c) 2022
# *** Copyright Notice ***
# “SOFTWARE NAME” was produced under the DOE Carbon Capture Simulation Initiative (CCSI), and is
# copyright (c) 2022 by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-Battelle, LLC, NOTRE
# DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
#
# *** License Agreement ***
#
# “SOFTWARE NAME” Copyright (c) 2022, by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-
# Battelle, LLC, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the
# following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided with the distribution.
# (3) Neither the name of the Carbon Capture Simulation for Industry Impact, TRIAD, LLNS, BERKELEY LAB,
# PNNL, UT-Battelle, LLC, ORNL, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, U.S. Dept. of Energy nor
# the names of its contributors may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features,
# functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory,
# without imposing a separate written license agreement for such Enhancements, then you hereby grant
# the following license: a non-exclusive, royalty-free perpetual license to install, use, modify, prepare
# derivative works, incorporate into other computer software, distribute, and sublicense such
# enhancements or derivative works thereof, in binary and source code form.
#
#################################################################################################################

# import libraries
from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
)

import pyomo.common.unittest as unittest

from pyomo.environ import *
from pyomo.dae import *

from itertools import permutations, product, combinations
import idaes

from fim_doe import *

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")



class doe_object_Tester(unittest.TestCase):
    ''' Test the kinetics example with both the sequential_finite mode and the direct_kaug mode 
    '''
    def setUP(self):
        import reactor_kinetics as reactor
        
        # define create model function 
        createmod = reactor.create_model
        
        # discretizer 
        disc = reactor.disc_for_measure
        
        # design variable and its control time set
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        dv_pass = {'CA0': [0],'T': t_control}

        # Define measurement time points
        t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        measure_pass = {'C':{'CA': t_measure, 'CB': t_measure, 'CC': t_measure}}
        measure_class =  Measurements(measure_pass)
        
        # Define parameter nominal value 
        parameter_dict = {'A1': 84.79085853498033, 'A2': 371.71773413976416, 'E1': 7.777032028026428, 'E2': 15.047135137500822}

        def generate_exp(t_set, CA0, T):  
            '''Generate experiments. 
            t_set: time control set for T.
            CA0: CA0 value
            T: A list of T 
            '''
            assert(len(t_set)==len(T)), 'T should have the same length as t_set'

            T_con_initial = {}
            for t, tim in enumerate(t_set):
                T_con_initial[tim] = T[t]

            dv_dict_overall = {'CA0': {0: CA0},'T': T_con_initial}
            return dv_dict_overall
        
        # empty prior
        prior_all = np.zeros((4,4))

        prior_pass=np.asarray(prior_all)
        
        ### Test sequential_finite mode
        exp1 = generate_exp(t_control, 5, [300, 300, 300, 300, 300, 300, 300, 300, 300])

        doe_object = DesignOfExperiments(parameter_dict, dv_pass,
                             measure_class, createmod,
                            prior_FIM=prior_pass, discretize_model=disc, args=[True])

    
        result = doe_object.compute_FIM(exp1,mode='sequential_finite', FIM_store_name = 'dynamic.csv', 
                                        store_output = 'store_output', read_output=None,
                                        scale_nominal_param_value=True, formula='central')


        result.calculate_FIM(doe_object.design_values)

        self.assertAlmostEqual(np.log10(result.trace), 2.962954, places=3)
        self.assertAlmostEqual(result.FIM[0][1], 1.840604, places=3)
        self.assertAlmostEqual(result.FIM[0][2], -70.238140, places=3)
        
        ### Test direct_kaug mode
        exp2 = generate_exp(t_control, 5, [570, 300, 300, 300, 300, 300, 300, 300, 300])
        
        doe_object2 = DesignOfExperiments(parameter_dict, dv_pass,
                             measure_class, createmod,
                            prior_FIM=prior_pass, discretize_model=disc, args=[False])
        result2 = doe_object2.compute_FIM(exp2,mode='direct_kaug', FIM_store_name = 'dynamic.csv', 
                                        store_output = 'store_output', read_output=None,
                                        scale_nominal_param_value=True, formula='central')
        
        result2.calculate_FIM(doe_object2.design_values)

        self.assertAlmostEqual(np.log10(result2.trace), 2.788587, places=3)
        self.assertAlmostEqual(np.log10(result2.det), 2.821840, places=3)
        self.assertAlmostEqual(np.log10(result2.min_eig), -1.012346, places=3)
        
            
        square_result, optimize_result= doe_object.optimize_doe(exp1, if_optimize=True, if_Cholesky=True,                                          scale_nominal_param_value=True, objective_option='det', 
                                                         L_initial=None)
        
        self.assertAlmostEqual(optimize_result.model.T[0], 477.134504, places=3)
        self.assertAlmostEqual(optimize_result.model.T[1], 300.000207, places=3)
        self.assertAlmostEqual(np.log10(optimize_result.trace), 2.982298, places=3)
        self.assertAlmostEqual(np.log10(optimize_result.det), 3.303190, places=3)
        

if __name__ == '__main__':
    unittest.main()





