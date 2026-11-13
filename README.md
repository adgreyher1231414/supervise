# Integrated Dynamic Life Cycle Assessment of UAV Last-Mile Logistics

Python reference **implementation of the stated method**, with an illustrative self-contained experiment.

## Run

```bash
python -m pip install -r requirements.txt
python run_experiments.py --output outputs --seed 2026 --mc 10
```

Outputs: `scenarios_144.csv`, `monte_carlo_results.csv` (1,440 rows), `results_144.csv`, sensitivity and ablation tables, seven publication-style demonstration PNG plots, and an ACO route demonstration JSON. `python -m unittest discover -s tests` runs unit tests.

## Implemented components

- Equations (hover power), (forward power) and (induced velocity); round-trip energy and wind perturbations.
- Battery NMC-811 retention `Q_n / Q_0 = 1 - 0.0018 sqrt(n) - 0.00042 sqrt(t_hours)`, 80% replacement rule.
- Per-mission battery manufacturing burden and one-time battery recycling credits; airframe impacts.
- Charging grid CI, fixed-rate fossil/H2 inputs, per-delivery life-cycle costing.
- Energy-based ACO routing with 80 ants / 500 iterations configurable, with depot return, payload cap, and 10% tightened energy-reserve test.
- Asymmetric 48+48+24+24 scenario design (4 zones x 3 routes x [4/4/2/2] replications), 10 wind Monte Carlo runs, 144-scenario Pareto identification, sensitivity and ablations.

## Critical reproduction limitations

This manuscript did not supply the actual 3,850 OSM routes, 2022–2023 hourly 47-zone ENTSO-E marginal emission factors, licensed ecoinvent 3.10 process inventories, published mission-level validation set, calibrated vehicle aerodynamic/phase parameters, manufacturers' purchase costs, nor the supplementary scenario list. **This repository does not pretend to reproduce those inputs.** It uses clearly editable `PLATFORMS`, `GRID_ZONES`, and `ROUTES_KM` DEMO assumptions. Its generated numbers **must not** be published as reproductions of the paper's figures or external MAE.

Key differences:

1. Fuel rates, hydrogen rates, platform drag/rotor properties, life-cycle purchase costs, motor efficiency and airframe manufacturing burden are placeholders. Replace with evidence-based inputs before scientific claims.
2. The route generator uses one representative distance per class with small jitter, rather than actual OSM spatial records. ACO demo solves a tiny single-UAV multi-stop route but is **not used** to create the 144 delivery route inputs.
3. Grid zones: four representative numeric CI values (68, 185, 239, 478 g/kWh) to reflect the paper's reported 36 low / 72 mixed / 36 high scenario counts. Not true hourly ENTSO-E data. `hourly_weighted_ci` is available for user-provided charging profiles.
4. The published corrected degradation equation and the cited ~1140 EOL cycles are **not necessarily mutually consistent** for a 5-year calendar life. With `t=43800 h`, the calendar term alone is ~0.0879 capacity loss, making EOL occur much earlier than 1140 cycles. The code follows the *equation*, not a forced replacement schedule.
5. The paper describes both **four grid-mix zones** and a **three-stratum** 36/72/36 table; this implementation uses two mixed zones. Individual zones are synthetic, as the detailed supplementary allocation is unavailable.
6. Cost/EOL parameters absent from the article are documented example inputs. The cross-platform outputs depend strongly on them.
7. A single-UAV capacity constraint is applied to the multi-stop ACO demonstration; published route-level formulation is otherwise underspecified for practical exact reconstruction.
8. Sensitivity elasticities here are finite-difference, and ablation MAE is **relative to our own full model**, not an external accuracy metric.

## How to use real data

- Populate platform entries in `idlca/core.py` with real aircraft properties, certified propulsion efficiencies, actual per-stage flight durations, purchased costs, material mass inventories and service lives.
- Replace `build_scenarios()` with a dataframe built from the provided OSM route geometries and matched ENTSO-E **marginal** hourly grid data (including cross-border attribution and 2022–2023 interpolation). Use `hourly_weighted_ci()` for multi-hour charging events.
- Substitute process GWP factors from your licensed ecoinvent/Brightway2 export and apply the declared ReCiPe 2016 Midpoint (H) GWP100 method. Validate impact-accounting conventions and allocation with the LCA practitioner.
- Calibrate the battery degradation coefficients with the **actual** NMC 811 experimental capacity curves; externally validate the energy model with flight telemetry.
- Recompute all plots and uncertainties after replacing assumptions, including independent inventory uncertainty if appropriate.

**Source:** uploaded manuscript titled *Integrated Dynamic Life Cycle Assessment of Economic and Carbon Footprint Impacts in UAV-Based Last-Mile Logistics*.
