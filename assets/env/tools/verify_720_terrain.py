"""Replay the placed aerial state with the production snow backend still enabled."""
import json,numpy as np,mujoco
from annotate_720 import model,ids,OUT
from flight_720 import poses,reference
from runtime import ROOT,SnowStepper
m,d=model();qa,va,names=ids(m);uu=np.array([i for i in range(m.nv) if i not in va]);s=SnowStepper(m,json.load(open(ROOT/'configs/g1_contacts.json')));a=np.load(OUT/'placed_aerial.npz');T=float(a['time'][-1]);d.qpos[:]=a['qpos'][0];d.qvel[:]=a['qvel'][0];ps=poses(m);M=np.zeros((m.nv,m.nv));maxforce=0;first=None;maxq=0
for k in range(round(T/m.opt.timestep)+1):
 t=k*m.opt.timestep;mujoco.mj_forward(m,d);target,vel,acc=reference(t,T,ps);mujoco.mj_fullM(m,M,d.qM)
 au=np.linalg.solve(M[np.ix_(uu,uu)],d.qfrc_passive[uu]-d.qfrc_bias[uu]-M[np.ix_(uu,va)]@acc)
 tau=M[np.ix_(va,va)]@acc+M[np.ix_(va,uu)]@au+d.qfrc_bias[va]-d.qfrc_passive[va]
 d.ctrl[:]=np.clip(target+(tau-m.actuator_biasprm[:,2]*vel)/m.actuator_gainprm[:,0],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
 if k%20==0:maxq=max(maxq,float(np.max(abs(d.qpos-a['qpos'][k//20]))))
 if t>=T:break
 s.step(d);cs=s.contacts(d);force=sum(max(c['force_contact_frame'][0],0) for c in cs)
 if force>1e-6 and first is None:first=t
 maxforce=max(maxforce,force)
r=dict(timestep_s=m.opt.timestep,duration_s=T,max_qpos_difference=float(maxq),maximum_airborne_snow_normal_force_N=float(maxforce),first_snow_force_time_s=first,warnings=d.warning.number.tolist(),passed=bool(maxq<1e-5 and maxforce<1e-5 and not any(d.warning.number)))
(OUT/'terrain_replay_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
