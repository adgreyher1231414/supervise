"""Reproducible, parameterized implementation of the manuscript's IDLCA equations.

The public manuscript lacks its underlying route/ENTSO-E/ecoinvent/cost inventories.
This module therefore distinguishes published constants from DEMONSTRATION assumptions.
All impacts returned here are illustrative, not independently validated paper results.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

G = 9.81
RHO = 1.225
BAT_EI = 12.6  # kg CO2e/kg, explicitly given in manuscript
EOL_BAT_EI = -2.3
EOL_CF_EI = 0.87
EOL_WEEE_EI = 0.34
ALPHA_C = 1.8e-3
ALPHA_CAL = 4.2e-4

@dataclass(frozen=True)
class Platform:
    name: str
    airframe_mass: float
    max_payload: float
    battery_wh: float
    battery_mass: float
    lifetime_missions: int
    # The following are EXPLICIT DEMONSTRATION assumptions not provided by the paper:
    disk_area: float
    cruise_speed: float
    frontal_area: float
    cd: float
    eta_motor: float
    eta_esc: float
    cruise_propulsion_factor: float  # VTOL wing-lift surrogate / hybrid correction
    hover_seconds: float
    climb_descent_wh: float
    purchase_usd: float
    battery_replacement_usd: float
    maintenance_usd: float
    regulator_usd: float
    eol_usd: float
    airframe_embodied_kg: float  # additional component manufacturing GWP / lifetime
    electricity_price: float = .22
    fuel_kg_per_km: float = 0.0
    fuel_emissions_kg_per_kg: float = 0.0
    fuel_usd_per_kg: float = 0.0

PLATFORMS = {
    'Multirotor': Platform('Multirotor',2.8,2.,480.,1.3,5000,.45,12.,.085,1.05,.90,.96,1.,100.,13.,5800.,300.,.30,.24,.04,260.,.22),
    'VTOL': Platform('VTOL',4.2,3.,720.,1.9,4800,.72,18.,.12,.8,.91,.96,.53,125.,20.,8500.,450.,.34,.26,.05,375.,.22),
    'Hydrogen FC': Platform('Hydrogen FC',5.6,4.,0.,0.,4200,.7,16.,.13,.9,.9,.96,1.,115.,16.,12300.,0.,.42,.3,.06,520.,.0,.00065,9.4,10.5),
    'Hybrid petrol': Platform('Hybrid petrol',6.1,5.,220.,.6,5200,.62,17.,.14,.95,.90,.96,.75,110.,15.,8700.,200.,.39,.26,.06,460.,.22,.0017,3.15,2.1),
}

# Four study zones with the manuscript's 36/72/36 grouping.
# 68, 212 and 478 g/kWh are from the paper; split of mixed zones is illustrative.
GRID_ZONES = {'low':68., 'mixed_A':185., 'mixed_B':239., 'high':478.}
ROUTES_KM = {'short':3., 'medium':8., 'long':14.}  # illustrative representatives


def hover_power_w(m_kg: float, disk_area: float, eta_motor: float, eta_esc: float,
                  rho: float=RHO) -> float:
    return (m_kg*G)**1.5 / math.sqrt(2*rho*disk_area) / (eta_motor*eta_esc)


def induced_velocity_ms(hover_power: float, disk_area: float, eta_motor: float,
                        eta_esc: float, rho: float=RHO) -> float:
    # Exactly follows the article's corrected Eq. (induced velocity).
    return math.sqrt(hover_power*eta_motor*eta_esc/(2*rho*disk_area))


def forward_power_w(p: Platform, mass_kg: float, airspeed_ms: float) -> float:
    ph = hover_power_w(mass_kg,p.disk_area,p.eta_motor,p.eta_esc)
    vi = induced_velocity_ms(ph,p.disk_area,p.eta_motor,p.eta_esc)
    x = max(.001,airspeed_ms)/(2*vi)
    return p.cruise_propulsion_factor*(ph/(x+math.sqrt(x*x+1)) +
                        .5*RHO*max(.001,airspeed_ms)**3*p.frontal_area*p.cd)


def mission_energy_kwh(p: Platform, one_way_km: float, payload_kg: float=1.5,
                       wind_ms: float=0., distance_overhead: float=.15,
                       charge_efficiency: float=.90) -> float:
    """Return grid-side equivalent electricity, a round-trip mission.

    Wind is the outgoing headwind; opposite on the empty return leg. Missing
    climb/descent phase timings are approximated by configurable energy overhead.
    """
    if payload_kg > p.max_payload: raise ValueError('Payload exceeds UAV rating')
    if one_way_km <= 0: raise ValueError('Route distance must be positive')
    distance_m = one_way_km*1000*(1+distance_overhead)
    takeoff_mass = p.airframe_mass + p.battery_mass + payload_kg
    return_mass = p.airframe_mass + p.battery_mass
    outbound_ground_speed = max(2.,p.cruise_speed-wind_ms)
    return_ground_speed = max(2.,p.cruise_speed+wind_ms)
    outbound_wh = forward_power_w(p,takeoff_mass,p.cruise_speed)*distance_m/outbound_ground_speed/3600
    return_wh = forward_power_w(p,return_mass,p.cruise_speed)*distance_m/return_ground_speed/3600
    hover_wh = (hover_power_w(takeoff_mass,p.disk_area,p.eta_motor,p.eta_esc) +
               hover_power_w(return_mass,p.disk_area,p.eta_motor,p.eta_esc))*p.hover_seconds/2/3600
    return (outbound_wh+return_wh+hover_wh+p.climb_descent_wh)/1000/charge_efficiency


def capacity_retention(n_cycles: float, calendar_hours: float, alpha_c: float=ALPHA_C,
                       alpha_cal: float=ALPHA_CAL) -> float:
    return max(0.,1-alpha_c*math.sqrt(max(0,n_cycles))-alpha_cal*math.sqrt(max(0,calendar_hours)))


def replacements_over_lifetime(p: Platform, energy_kwh: float, dod: float=.7,
                               years: float=5., alpha_c: float=ALPHA_C,
                               alpha_cal: float=ALPHA_CAL) -> tuple[float,float,float]:
    """Compute replacement count by sequential capacity-loss simulation.

    Returns (# replacements, mean cycles per mission, end retention). These
    replacement counts are discrete; paper's quoted 2.1 etc are estimated means.
    """
    if not p.battery_wh: return 0.,0.,1.
    cycles_per_mission = energy_kwh/(p.battery_wh/1000*dod)
    cyc=0.; hours=0.; replacement_count=0
    h_per_mission=years*365.25*24/p.lifetime_missions
    for _ in range(p.lifetime_missions):
        cyc += cycles_per_mission; hours += h_per_mission
        if capacity_retention(cyc,hours,alpha_c,alpha_cal)<.8:
            replacement_count += 1; cyc=0.; hours=0.
    return float(replacement_count),cycles_per_mission,capacity_retention(cyc,hours,alpha_c,alpha_cal)


def hourly_weighted_ci(start_hour: int, charging_kwh: float, hourly_ci: np.ndarray,
                       charging_kw: float=1.) -> float:
    """Energy-weighted CI for a charging window crossing hourly boundaries."""
    remaining=float(charging_kwh); hour=int(start_hour); total=0.
    if remaining <= 0: return 0.
    if charging_kw <= 0: raise ValueError('charging_kw must be positive')
    while remaining>1e-12:
        e=min(remaining,charging_kw)
        total+=e*float(hourly_ci[hour%len(hourly_ci)])
        remaining-=e; hour+=1
    return total/charging_kwh


def assess(p: Platform, distance_km: float, ci_g_per_kwh: float,
           wind_ms: float=0., payload_kg: float=1.5,
           distance_overhead: float=.15, battery_ei: float=BAT_EI,
           include_battery: bool=True, include_eol: bool=True,
           green_hydrogen: bool=False, alpha_c: float=ALPHA_C,
           lifetime_multiplier: float=1., infrastructure_gwp: float=.01) -> dict:
    """Per-delivery cradle-to-grave accounting with transparently assumed inputs."""
    e_kwh=mission_energy_kwh(p,distance_km,payload_kg,wind_ms,distance_overhead)
    lifetime=max(1,round(p.lifetime_missions*lifetime_multiplier))
    # If lifetime changes, maintain approximate same utilization rate (years scale).
    from dataclasses import replace
    ps=replace(p,lifetime_missions=lifetime)
    reps,cyc,ret=replacements_over_lifetime(ps,e_kwh,alpha_c=alpha_c,years=5*lifetime_multiplier)
    # Electric vehicles use grid energy. For H2 / petrol the propulsive energy
    # proxy is tracked but not treated as purchased electricity.
    grid_kwh=e_kwh if p.name in ('Multirotor','VTOL') else (e_kwh*.05 if p.name=='Hybrid petrol' else 0.)
    gwp_energy=grid_kwh*ci_g_per_kwh/1000
    fuel_kg = p.fuel_kg_per_km * distance_km*2*(1+distance_overhead)
    gwp_fuel = fuel_kg*(.9 if green_hydrogen and p.name=='Hydrogen FC' else p.fuel_emissions_kg_per_kg)
    gwp_operation=gwp_energy+gwp_fuel
    gwp_airframe=p.airframe_embodied_kg/lifetime
    # New first battery plus the replacement packs; one-time accounting.
    gwp_battery=((1+reps)*p.battery_mass*battery_ei/lifetime) if include_battery else 0.
    # Demonstration airframe fractions; actual ecoinvent inventory is missing.
    airframe_eol=(p.airframe_mass*.38*EOL_CF_EI+p.airframe_mass*.15*EOL_WEEE_EI)/lifetime
    battery_eol=(-(1+reps)*p.battery_mass*2.3/lifetime) if include_battery else 0.
    gwp_eol=(airframe_eol+battery_eol) if include_eol else 0.
    gwp_infra=infrastructure_gwp
    gwp_total=gwp_operation+gwp_airframe+gwp_battery+gwp_eol+gwp_infra
    cost_capital=p.purchase_usd/lifetime
    cost_repl=reps*p.battery_replacement_usd/lifetime if include_battery else 0.
    cost_energy=grid_kwh*p.electricity_price+fuel_kg*p.fuel_usd_per_kg
    cost_eol=p.eol_usd if include_eol else 0.
    cost_total=cost_capital+cost_repl+cost_energy+p.maintenance_usd+p.regulator_usd+cost_eol
    return {'energy_kwh':e_kwh,'grid_kwh':grid_kwh,'fuel_kg':fuel_kg,
            'battery_replacements':reps,'full_equiv_cycles_per_mission':cyc,
            'battery_remaining_capacity':ret, 'gwp_operation':gwp_operation,
            'gwp_airframe':gwp_airframe,'gwp_battery':gwp_battery,
            'gwp_eol':gwp_eol,'gwp_infrastructure':gwp_infra,
            'gwp_total':gwp_total,'cost_capital':cost_capital,
            'cost_battery':cost_repl,'cost_energy':cost_energy,
            'cost_maintenance':p.maintenance_usd,'cost_regulatory':p.regulator_usd,
            'cost_eol':cost_eol,'cost_total':cost_total}


def non_dominated_mask(costs: np.ndarray, emissions: np.ndarray) -> np.ndarray:
    """Both objectives minimize. At least one strictly better to dominate."""
    c=np.asarray(costs,float); g=np.asarray(emissions,float)
    mask=np.ones(len(c),dtype=bool)
    for i in range(len(c)):
        mask[i]=not np.any((c<=c[i]) & (g<=g[i]) & ((c<c[i])|(g<g[i])))
    return mask


def optimize_route_aco(points: np.ndarray, platform: Platform, payloads: np.ndarray|None=None,
                       n_ants: int=80, iterations: int=500, evaporation: float=.12,
                       pheromone_q: float=150., seed: int=42, capacity_wh: float|None=None,
                       early_stop: int=50, wind_ms: float=0.) -> dict:
    """Single-UAV, single-tour ACO with roundtrip reserve feasibility.

    points: Nx2 kilometers, point 0 is depot; total payload must fit. Each
    delivery decreases payload. Returns an energy-minimizing candidate route.
    This is a compact ACO demonstration, not verification of the archived study.
    """
    points=np.asarray(points,float); n=len(points)
    if n<2: raise ValueError('Need depot and at least one stop')
    if payloads is None: payloads=np.full(n-1,1.5)
    payloads=np.asarray(payloads,float)
    if len(payloads)!=n-1 or payloads.sum()>platform.max_payload+1e-9:
        raise ValueError('Single-flight total payload exceeds capacity; split into separate tours')
    dist=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)*1.15
    rng=np.random.default_rng(seed); pher=np.ones((n,n),dtype=float)
    np.fill_diagonal(pher,0.)
    cap=platform.battery_wh if capacity_wh is None else capacity_wh
    def tour_energy(tour):
        load=float(payloads.sum()); total=0.
        for a,b in zip(tour[:-1],tour[1:]):
            m=platform.airframe_mass+platform.battery_mass+load
            power=forward_power_w(platform,m,platform.cruise_speed)
            groundspeed=max(2,platform.cruise_speed-wind_ms)
            total+=power*(dist[a,b]*1000/groundspeed)/3600
            if b!=0: load-=payloads[b-1]
        total+=platform.climb_descent_wh+platform.hover_seconds*hover_power_w(
            platform.airframe_mass+platform.battery_mass,
            platform.disk_area,platform.eta_motor,platform.eta_esc)/3600
        return total/.9
    best=None;best_e=float('inf'); stagnant=0
    for it in range(iterations):
        samples=[]
        for _ in range(n_ants):
            remaining=set(range(1,n)); tour=[0]
            while remaining:
                i=tour[-1]; cand=sorted(remaining)
                attractiveness=np.array([pher[i,j]**1.0/(dist[i,j]+1e-6)**2.0 for j in cand])
                probs=attractiveness/attractiveness.sum()
                j=int(rng.choice(cand,p=probs));tour.append(j);remaining.remove(j)
            tour.append(0);energy=tour_energy(tour)
            if cap is None or energy <= .9*cap: samples.append((tour,energy))
            if energy<best_e and (cap is None or energy <= .9*cap):
                improvement=(best_e-energy)/max(1,best_e) if math.isfinite(best_e) else float("inf")
                best_e=energy;best=tour
                stagnant=stagnant+1 if improvement<.001 else 0
        pher*=(1-evaporation)
        for tour,energy in samples:
            increment=pheromone_q/max(energy,1e-9)
            for i,j in zip(tour[:-1],tour[1:]): pher[i,j]+=increment
        if stagnant>=early_stop: break
    if best is None: return {'feasible':False,'route':None,'energy_wh':None,'iterations':it+1}
    return {'feasible':True,'route':best,'energy_wh':best_e,'iterations':it+1}
