"""Place the native aerial solution by Galilean boost; test uncontrolled fixed-pose landing.
Run from any cwd after flight_720.py / shoot_cross_720.py.
"""
import sys,json,math,numpy as np,mujoco
from scipy.optimize import brentq

from annotate_720 import model,ids,OUT
from runtime import SnowStepper,ROOT,Terrain
from flight_720 import poses
m,d=model();qa,va,names=ids(m);s=SnowStepper(m,json.load(open(ROOT/'configs/g1_contacts.json')));surf=s.surface;tr=Terrain('tabletop_steep');a=np.load(OUT/'aerial_shooting_probe.npz');T=a['time'][-1];v0=np.array([6*math.cos(math.radians(18)),0,6*math.sin(math.radians(18))+1.2])
def points(q):
 d.qpos[:]=q;mujoco.mj_forward(m,d);return d.xpos[surf.bids]+np.einsum('nij,nj->ni',d.xmat[surf.bids].reshape(-1,3,3),surf.local)
def zclear(q):
 p=points(q);return float(np.min(p[:,2]-surf.radii-tr.height(p[:,0],p[:,1])))
q0=a['qpos'][0].copy();dz=.002-zclear(q0)
def shifted(k,v):
 q=a['qpos'][k].copy();q[:3]+=(np.array([v,0,v*math.tan(math.radians(18))])-v0)*a['time'][k];q[2]+=dz;return q
v=brentq(lambda v:zclear(shifted(-1,v)),2,12);boost=np.array([v,0,v*math.tan(math.radians(18))])-v0
Q=np.array([shifted(k,v) for k in range(len(a['time']))]);V=a['qvel'].copy();V[:,:3]+=boost
clear=np.array([zclear(q) for q in Q]);print('PLACE',v,v/math.cos(math.radians(18)),dz,'clear',clear.min(),'x',Q[[0,-1],0],flush=True)
np.savez_compressed(OUT/'placed_aerial.npz',**{k:a[k] for k in a.files if k not in ['qpos','qvel','com']},qpos=Q,qvel=V,com=a['com']+a['time'][:,None]*boost+np.array([0,0,dz]),vertical_clearance=clear)
(OUT/'placement.json').write_text(json.dumps(dict(initial_speed_m_s=v/math.cos(math.radians(18)),initial_com_velocity=[v,0,v*math.tan(math.radians(18))],launch_root=Q[0,:3].tolist(),landing_root=Q[-1,:3].tolist(),minimum_interior_vertical_clearance_m=float(clear[1:-1].min()),final_vertical_clearance_m=float(clear[-1]),initial_vertical_clearance_m=float(clear[0]),T=float(T)),indent=2))
d.qpos[:]=Q[-1];d.qvel[:]=V[-1];d.time=T;land=poses(m)[2];rows=[];mincon=0;peak=np.zeros(29)
for k in range(2001):
 if k%20==0:
  mujoco.mj_forward(m,d);rows.append(dict(time=d.time,qpos=d.qpos.copy(),qvel=d.qvel.copy(),up=d.xmat[1].reshape(3,3)[2,2]))
 d.ctrl[:]=land;s.step(d);peak=np.maximum(peak,abs(d.qfrc_actuator[va]));mincon=min(mincon,min([c.dist for c in d.contact[:d.ncon] if not c.exclude] or [0]))
np.savez_compressed(OUT/'landing_probe.npz',**{key:np.array([r[key] for r in rows]) for key in rows[0]})
print('LAND',mincon,min(r['up'] for r in rows),rows[-1]['up'],d.warning.number.tolist(),flush=True)
(OUT/'landing_probe.json').write_text(json.dumps(dict(duration_s=.5,minimum_contact_distance_m=float(mincon),min_up=float(min(r['up'] for r in rows)),final_up=float(rows[-1]['up']),warning_counts=d.warning.number.tolist(),actual_peak_joint_torque_Nm=peak.tolist()),indent=2))
