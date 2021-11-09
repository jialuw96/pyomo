#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available
from pyomo.environ import (
    Var, ConcreteModel, Reals, ExternalFunction,
    Objective, Constraint, sqrt, sin, SolverFactory
    )

logger = logging.getLogger('pyomo.contrib.trustregion')

@unittest.skipIf(not SolverFactory('ipopt').available(False), "The IPOPT solver is not available")
@unittest.skipIf(not SolverFactory('gjh').available(False), "The GJH solver is not available")
@unittest.skipIf(not numpy_available, "Cannot test the trustregion solver without numpy")
class TestTrustRegionConfig(unittest.TestCase):

    def setUp(self):

        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(2), initialize=2.)
        m.x[1] = 1.0

        def blackbox(a,b):
            return sin(a-b)
        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
                + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
            )
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + self.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))

        self.m = m.clone()

    def try_solve(self):
        status = True
        try:
            self.TRF.solve(self.m, [self.bb], **kwds)
        except Exception as e:
            print('error calling optTRF.solve: %s' % str(e))
            status = False
        return status

    def test_config_vars(self):
        self.TRF = SolverFactory('trustregion')
        self.assertEqual(self.TRF.config.trust_radius, 1.0)

        # Both persistent and local values should be 1.0
        solve_status = self.try_solve()
        self.assertTrue(solve_status)
        self.assertEqual(self.TRF.config.trust_radius, 1.0)
        self.assertEqual(self.TRF._local_config.trust_radius, 1.0)
