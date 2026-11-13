"""Run IDLCA illustrative 144-scenario experiment and export figures/tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from idlca.core import (PLATFORMS, GRID_ZONES, ROUTES_KM, assess,
                        non_dominated_mask, capacity_retention, optimize_route_aco)


def build_scenarios(seed=2026):
    rng=np.random.default_rng(seed)
    rows=[]
    for name,p in PLATFORMS.items():
        repeats=4 if name in ('Multirotor','VTOL') else 2
        for route, d in ROUTES_KM.items():
            for zone,ci in GRID_ZONES.items():
                for rep in range(repeats):
                    # Route and wind samples are illustrative, not OSM/ENTSO-E observations
                    rows.append(dict(platform=name,route=route,zone=zone,
                                     ci=ci, replicate=rep+1,
                                     distance_km=d * rng.uniform(.92,1.08),
                                     base_wind_ms=rng.uniform(-2,2)))
    return pd.DataFrame(rows)


def plot_scenario_outputs(results, outdir):
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(route,sub) in zip(axes,results.groupby('route',sort=False)):
        for name,g in sub.groupby('platform',sort=False):
            ax.scatter(g.ci,g.gwp_total,s=17,alpha=.6,label=name)
        ax.set(title=route.capitalize(),xlabel='Grid carbon intensity (g CO2e/kWh)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Illustrative GWP (kg CO2e/delivery)')
    axes[-1].legend(fontsize=7,loc='best')
    fig.tight_layout();fig.savefig(outdir/'fig2_gwp_comparison_demo.png',dpi=180);plt.close(fig)

    phase_cols=['gwp_operation','gwp_airframe','gwp_battery','gwp_eol','gwp_infrastructure']
    means=results.groupby('platform',sort=False)[phase_cols].mean()
    fig,ax=plt.subplots(figsize=(9,4))
    bottom=np.zeros(len(means))
    for col in phase_cols:
        vals=means[col].values
        ax.bar(means.index,vals,bottom=bottom,label=col.replace('gwp_',''))
        bottom+=vals
    ax.set(ylabel='Illustrative GWP (kg CO2e/delivery)',title='Life-cycle contributions (with recycling credits)')
    ax.legend(ncol=3,fontsize=8);fig.tight_layout()
    fig.savefig(outdir/'fig4_phase_contribution_demo.png',dpi=180);plt.close(fig)

    components=['cost_capital','cost_battery','cost_energy','cost_maintenance','cost_regulatory','cost_eol']
    costs=results.groupby('platform',sort=False)[components].mean()
    fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(len(costs))
    for col in components:
        ax.bar(costs.index,costs[col].values,bottom=bottom,label=col.replace('cost_',''))
        bottom+=costs[col].values
    ax.set(ylabel='Illustrative cost (USD/delivery)',title='Per-delivery cost decomposition')
    ax.legend(ncol=3,fontsize=8);fig.tight_layout()
    fig.savefig(outdir/'fig3_cost_breakdown_demo.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in results.groupby('platform',sort=False):
        ax.scatter(g.gwp_total,g.cost_total,s=19,alpha=.58,label=name)
    nd=results[results.pareto]
    nd=nd.sort_values('gwp_total')
    ax.plot(nd.gwp_total,nd.cost_total,'k--',label='Verified non-dominated')
    ax.set(xlabel='Illustrative GWP (kg CO2e/delivery)',ylabel='Illustrative cost (USD/delivery)',title='Cost–carbon Pareto frontier')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(outdir/'fig7_pareto_frontier_demo.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    cycles=np.linspace(0,1500,300)
    for h,label in [(0,'No calendar ageing'),(8760,'1 calendar year'),(43800,'5 calendar years')]:
        ax.plot(cycles,[capacity_retention(n,h) for n in cycles],label=label)
    ax.axhline(.8,ls='--',c='k',lw=.9,label='80% EOL')
    ax.set(xlabel='Full equivalent cycles',ylabel='Capacity retention',title='Published capacity-fade equation')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(outdir/'fig8_degradation_model_demo.png',dpi=180);plt.close(fig)


def sensitivity(output_dir):
    p=PLATFORMS['Multirotor']
    base=dict(distance_km=8.,ci_g_per_kwh=212.,payload_kg=1.5)
    changes={'grid_CI':('ci_g_per_kwh',212.),'battery_EI':('battery_ei',12.6),
             'distance':('distance_km',8.),'payload':('payload_kg',1.5),
             'service_life':('lifetime_multiplier',1.),'infrastructure':('infrastructure_gwp',.01)}
    output=[]; y0=assess(p,**base)['gwp_total']
    for label,(param,val) in changes.items():
        vals=[]
        for factor in [.8,1.2]:
            kw=base.copy();kw[param]=val*factor
            vals.append(assess(p,**kw)['gwp_total'])
        # central finite-difference elasticity
        elast=(vals[1]-vals[0])/(.4*y0)
        output.append(dict(parameter=label,minus20_change_pct=(vals[0]/y0-1)*100,
                           plus20_change_pct=(vals[1]/y0-1)*100,elasticity=elast))
    df=pd.DataFrame(output);df.to_csv(output_dir/'sensitivity.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ypos=np.arange(len(df))
    ax.barh(ypos-.18,df.minus20_change_pct,height=.34,label='-20%')
    ax.barh(ypos+.18,df.plus20_change_pct,height=.34,label='+20%')
    ax.set_yticks(ypos,df.parameter)
    ax.axvline(0,c='k',lw=.7)
    ax.set(xlabel='Illustrative change in GWP (%)',title='OAT sensitivity (example assumptions)')
    ax.legend();fig.tight_layout();fig.savefig(output_dir/'fig5_sensitivity_analysis_demo.png',dpi=180);plt.close(fig)
    return df


def ablations(scenarios,outdir):
    rows=[]
    for _,s in scenarios.iloc[:21].iterrows():
        p=PLATFORMS[s.platform]
        kw=dict(distance_km=float(s.distance_km),ci_g_per_kwh=float(s.ci),wind_ms=float(s.base_wind_ms))
        full=assess(p,**kw)['gwp_total']
        variants={
            'no_battery':assess(p,**kw,include_battery=False)['gwp_total'],
            'static_grid':assess(p,**{**kw,'ci_g_per_kwh':212.})['gwp_total'],
            'no_route_overhead':assess(p,**kw,distance_overhead=0.)['gwp_total'],
            'no_EOL':assess(p,**kw,include_eol=False)['gwp_total'],
            'static_all':assess(p,**{**kw,'ci_g_per_kwh':212.},include_battery=False,
                                include_eol=False,distance_overhead=0.)['gwp_total']}
        for name,value in variants.items():rows.append(dict(variant=name,absolute_change=abs(value-full)))
    summary=pd.DataFrame(rows).groupby('variant').absolute_change.mean().reset_index()
    summary.to_csv(outdir/'ablation.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(summary.variant,summary.absolute_change)
    ax.set(ylabel='MAE relative to full model (kg CO2e)',title='Module-removal ablation (not external validation)')
    fig.tight_layout();fig.savefig(outdir/'fig6_ablation_modules_demo.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='outputs')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--mc',type=int,default=10)
    args=ap.parse_args()
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    scenarios=build_scenarios(args.seed)
    assert len(scenarios)==144
    scenarios.to_csv(outdir/'scenarios_144.csv',index=False)
    rows=[]
    for scenario_id,s in scenarios.iterrows():
        p=PLATFORMS[s.platform]
        for mc_seed in range(args.mc):
            wind=float(s.base_wind_ms+rng.uniform(-.15,.15)*max(1.,abs(s.base_wind_ms)))
            payload=1.5*rng.uniform(.9,1.1)
            values=assess(p,float(s.distance_km),float(s.ci),wind_ms=wind,payload_kg=payload)
            rows.append(dict(scenario_id=scenario_id,platform=s.platform,route=s.route,
                             zone=s.zone,ci=s.ci,replicate=s.replicate,mc_seed=mc_seed,
                             distance_km=s.distance_km,wind_ms=wind,payload_kg=payload,**values))
    monte_carlo=pd.DataFrame(rows)
    monte_carlo.to_csv(outdir/'monte_carlo_results.csv',index=False)
    numeric=monte_carlo.select_dtypes(include='number').columns.difference(['scenario_id','mc_seed','replicate'])
    summary=monte_carlo.groupby('scenario_id',as_index=False)[list(numeric)].mean()
    metadata=scenarios[['platform','route','zone','replicate']].copy()
    metadata.insert(0,'scenario_id',metadata.index)
    summary=metadata.merge(summary,on='scenario_id',how='left')
    summary['pareto']=non_dominated_mask(summary.cost_total.values,summary.gwp_total.values)
    summary.to_csv(outdir/'results_144.csv',index=False)
    plot_scenario_outputs(summary,outdir)
    sensitivity(outdir)
    ablations(scenarios,outdir)
    # A small multi-stop example: three payloads together meet capacity limit
    demo_points=np.array([[0,0],[1.2,.8],[-.8,1.3],[.6,1.9]])
    import json
    demo=optimize_route_aco(demo_points,PLATFORMS['VTOL'],payloads=np.array([.7,.8,.6]),
                            n_ants=30,iterations=100)
    (outdir/'aco_route_demo.json').write_text(json.dumps(demo,indent=2))
    print('Created',len(scenarios),'scenarios and',len(monte_carlo),'Monte Carlo rows')
    print('GWP range',round(summary.gwp_total.min(),4),round(summary.gwp_total.max(),4),'kg/delivery')
    print('Cost range',round(summary.cost_total.min(),4),round(summary.cost_total.max(),4),'USD/delivery')
    print('These are SIMULATED demo values, not the reported measured results.')
    print('Results saved to',outdir.resolve())

if __name__=='__main__':main()
