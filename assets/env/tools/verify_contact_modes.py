"""Contact-mode regression: true external work, finite grip, inversion and release."""
import json
import numpy as np
import mujoco
import audit_contact_details as details
from audit_dynamics import ROOT,make,run,snapshot
OUT=ROOT/'reports/repairs_v1_2';OUT.mkdir(exist_ok=True);details.OUT=OUT
R={'checks':[],'measurements':{}}
def save():
 R['passed']=all(c['passed'] for c in R['checks']);(OUT/'contact_modes.json').write_text(json.dumps(R,indent=2)+'\n')
def check(name,ok,**kw):R['checks'].append(dict(name=name,passed=bool(ok),**kw));save();print(name,bool(ok),flush=True)
for roll,fy,hold in [(0,5,True),(0,25,False),(35,25,True),(35,100,False),(35,150,False)]:
 m,d,s=make(roll=roll,payload_inertia=True);run(m,d,s,.8);r=run(m,d,s,.4,force=np.array([0.,fy,0.]));v=float(d.qvel[1]);check(f'grip_{roll}_{fy}',abs(v)<.02 if hold else abs(v)>.05,final_vy=v,work_J=r['applied_work_J'])
for hf in [False,True]:
 m,d,s=make(flat_hfield=hf);run(m,d,s,.3);d.qvel[0]=4;angle=0;fmax=0
 for _ in range(400):
  s.step(d)
  for c in s.contacts(d):
   if c['force_contact_frame'][0]>.01:
    angle=max(angle,float(np.degrees(np.arccos(np.clip(c['frame'][2],-1,1)))))
  f,_,_,_=snapshot(m,d,s);fmax=max(fmax,abs(float(f[1])))
 check(f'flat_normals_{hf}',angle<.01 and fmax<1,normal_error_deg=angle,max_Fy_N=fmax)
# Geometric inversion: upper-skin samples, actual ground support, no high edge grip.
m,d,s=make(pitch=180,payload_inertia=True);d.qpos[2]=.12
s.prepare(d);check('inverted_starts_clear',len(s.contacts(d))==0)
r=run(m,d,s,.8);s.prepare(d)
mus=[c.friction[1] for c in d.contact[:d.ncon] if not c.exclude and s.snow in c.geom]
check('inverted_upper_skin',r['minimum_contact_distance_m']>-.025 and len(mus)>0 and max(mus)<=.080001,sink_m=r['minimum_contact_distance_m'],max_mu=float(max(mus)) if mus else None)
a=details.release_and_pitch();R['measurements']['release']=a
check('airborne_zero_force',all(v['force_peak']==0 and v['no_contact_steps']==v['steps'] for v in a['airborne']))
check('pitch_does_not_activate_edge',a['pure_pitch_max_mu_lateral'] is not None and a['pure_pitch_max_mu_lateral']<=.080001)
t=details.transfer();R['measurements']['transfer']=t
for r in t:check('edge_transfer_'+str(r['dt']),r['maximum_energy_above_initial_plus_work_J']<.05 and r['min_distance_m']>-.005 and max(r['warnings'])==0,**r)
save()
if not R['passed']:raise SystemExit(1)
