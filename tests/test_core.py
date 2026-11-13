import unittest
import numpy as np
from idlca.core import (PLATFORMS,hover_power_w,induced_velocity_ms,capacity_retention,
                        mission_energy_kwh,assess,non_dominated_mask,optimize_route_aco)

class TestCore(unittest.TestCase):
    def test_physics(self):
        p=PLATFORMS['Multirotor'];power=hover_power_w(5.6,p.disk_area,p.eta_motor,p.eta_esc)
        self.assertGreater(power,0)
        self.assertGreater(induced_velocity_ms(power,p.disk_area,p.eta_motor,p.eta_esc),0)
    def test_degradation(self):
        self.assertEqual(capacity_retention(0,0),1.)
        self.assertLess(capacity_retention(1200,43800),.9)
    def test_energy_increases_with_route(self):
        p=PLATFORMS['Multirotor']
        self.assertLess(mission_energy_kwh(p,3),mission_energy_kwh(p,8))
    def test_totals(self):
        r=assess(PLATFORMS['Multirotor'],8,212)
        self.assertAlmostEqual(r['gwp_total'],sum(r[k] for k in ('gwp_operation','gwp_airframe','gwp_battery','gwp_eol','gwp_infrastructure')))
        self.assertAlmostEqual(r['cost_total'],sum(r[k] for k in ('cost_capital','cost_battery','cost_energy','cost_maintenance','cost_regulatory','cost_eol')))
    def test_pareto(self):
        np.testing.assert_array_equal(non_dominated_mask([1,2,3,1],[3,2,1,4]),[True,True,True,False])
    def test_aco(self):
        a=optimize_route_aco(np.array([[0,0],[.4,.4],[.3,.6]]),PLATFORMS['VTOL'],np.array([.4,.3]),n_ants=8,iterations=15)
        self.assertTrue(a['feasible']);self.assertEqual(set(a['route']),{0,1,2})

if __name__=='__main__':unittest.main()
