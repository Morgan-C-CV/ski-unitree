"""Summarize measured audit gates without relabeling untested capabilities as passed."""
import json
import math
from pathlib import Path
import numpy as np
from runtime import ROOT

def main():
    out=ROOT/'reports/dynamics_audit';audit=json.loads((out/'audit.json').read_text());details=json.loads((out/'contact_details.json').read_text());struct=json.loads((out/'structure_flex.json').read_text())
    norms=details['flat_heightfield_contact_normals'];plane=norms[0];hf=norms[1]
    speed_error=abs(hf['final_vx']-plane['final_vx'])/abs(plane['final_vx'])
    turns=[]
    for dt in [.001,.0005]:
        a=np.genfromtxt(out/f'carve_35_{dt}.csv',delimiter=',',names=True);i=np.argmin(abs(a['time']-1.8));vx=float(a['vx'][i]);vy=float(a['vy'][i]);turns.append({'dt':dt,'sample_time_s':float(a['time'][i]),'elapsed_glide_s':float(a['time'][i]-.8),'speed_m_s':math.hypot(vx,vy),'motion_direction_deg':math.degrees(math.atan2(vy,vx)),'position_xy_m':[float(a['x'][i]),float(a['y'][i])]})
    turn_error=abs(turns[0]['speed_m_s']-turns[1]['speed_m_s'])/turns[1]['speed_m_s']
    categories=[
        {'id':'geometry','status':'passed_limited','evidence':'42 MJCF files compile, source geometry/inertia available; earlier random terrain height/query tests pass. This does not certify collision normals.'},
        {'id':'base_static','status':'passed_fixture','evidence':'Static support balances 188.85 N; base ~50%, edges ~25% each when flat.'},
        {'id':'elasticity','status':'passed_discrete_model','evidence':'5/10/20 N cantilever loads within 0.64% of configured small-angle discrete stiffness; unload residual <1e-9 rad. Real EI not calibrated.'},
        {'id':'edge_static','status':'passed_fixture','evidence':'At 35deg loaded edge carries support; 100N transverse force held, 150N slips; pure pitch does not activate edging.'},
        {'id':'edge_dynamic','status':'partial','evidence':'Slow controlled transfer has small energy residual after recorded actuator work, but ~262N one-step force changes; high-speed transfer unvalidated.'},
        {'id':'heightfield_collision','status':'failed','evidence':f'Flat heightfield normal errors reach {hf["max_normal_error_deg"]:.2f}deg and speed differs {speed_error:.2%} from plane; finer grid does not converge.'},
        {'id':'carving','status':'failed_convergence','evidence':f'35deg roll, 4m/s initial speed, zero yaw drive: speed differs {turn_error:.2%} after ~1s between 1ms and 0.5ms; trajectories diverge.'},
        {'id':'g1_self_collision','status':'failed','evidence':'Assembly contype=4/conaffinity=3 blocks robot-robot pairs; no explicit pairs restore them.'},
        {'id':'flat_impact','status':'passed_limited','evidence':'First-contact 80ms impulse window converges across 1/0.5/0.25ms, but this is flat, orientation-clamped, non-robot landing.'},
        {'id':'high_course_dynamics','status':'failed_or_unverified','evidence':'Mogul probes retain centimetre penetration after payload-inertia correction; jump trajectories remain step-sensitive. No full G1 course closure.'},
        {'id':'RL_wrapper','status':'incomplete','evidence':'CPU stepping and terrain query adapters exist; reset/observation/action/phase/reward/termination schema and batch backend integration are not completed.'}]
    verdict={'overall':'NOT_ACCEPTED_FOR_FORMAL_RL','training_ready':False,'production_physics_modified':False,'categories':categories,'flat_heightfield_speed_relative_error':speed_error,'carve_speed_dt_relative_error':turn_error,'carve_comparison_at_one_second':turns,'blocking_ids':['heightfield_collision','carving','g1_self_collision','high_course_dynamics'],'priority_order':['Restore and test original G1 self-collision filtering.','Redesign/validate the base proxy-heightfield collision combination; edge-only ablation is diagnostic, not a replacement ski.','Repeat fixed-roll/free-yaw tests across dt, segment count, contact density, load and speed before claiming carving.','Validate high-terrain impact/flight/recontact with well-defined fixtures, then G1 closed-loop tasks.','Complete reset/observations/commands/reward/termination/phase and backend-specific RL integration.']}
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps({'overall':verdict['overall'],'plane_vs_hfield_speed_error':speed_error,'carving_dt_speed_error':turn_error,'turns':turns},indent=2))

if __name__=='__main__':main()
