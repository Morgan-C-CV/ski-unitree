"""Crossed-ski upright 720: geometry-constrained key poses on official G1."""
import json,math
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from annotate_720 import model,ids,OUT

def grab_points(m,d):
 b=m.body('right_wrist_yaw_link').id;p=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.10,-.003,-.018])
 b=m.body('right_ski_segment_1').id;g=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([-.05,-.045,.010])
 return p,g

def solve_cross():
 m,d=model();qa,va,names=ids(m);d.qpos[:3]=[0,0,30];d.qpos[3:7]=[1,0,0,0];nom=np.zeros(29)
 for side,sign in [('left',1),('right',-1)]:
  for key,value in [('hip_pitch_joint',-1.55),('knee_joint',2.4),('ankle_pitch_joint',-.85),('hip_yaw_joint',sign*.7),('hip_roll_joint',sign*.12),('ankle_roll_joint',-sign*.12),('shoulder_roll_joint',sign*.3),('elbow_joint',.6)]:nom[names.index(side+'_'+key)]=value
 nom[names.index('right_shoulder_pitch_joint')]=.3;nom[names.index('waist_yaw_joint')]=-.25
 ji=m.actuator_trnid[:,0];lo=m.jnt_range[ji,0]+.025;hi=m.jnt_range[ji,1]-.025
 lo[names.index('waist_pitch_joint')]=-.4;hi[names.index('waist_pitch_joint')]=.4
 lo[names.index('waist_yaw_joint')]=-.6;hi[names.index('waist_yaw_joint')]=.6
 lo[names.index('waist_roll_joint')]=-.25;hi[names.index('waist_roll_joint')]=.25
 # Let the contact target choose a feasible compact crouch, not an arbitrary skeleton.
 variable=[i for i,n in enumerate(names) if not n.startswith('left_shoulder') and not n.startswith('left_elbow') and not n.startswith('left_wrist')]
 ski_ids=[m.body(side+'_ski_segment_3').id for side in ['left','right']]
 desired=[Rotation.from_euler('ZY',[-45,10],degrees=True).as_matrix(),Rotation.from_euler('ZY',[45,35],degrees=True).as_matrix()]
 def evaluate(x,return_q=False):
  q=nom.copy();q[variable]=x;d.qpos[qa]=q;mujoco.mj_forward(m,d);res=[]
  centers=[d.xpos[b]-d.qpos[:3] for b in ski_ids]
  ax=[d.xmat[b].reshape(3,3)[:2,0] for b in ski_ids];res.append(float(ax[0]@ax[1]/(np.linalg.norm(ax[0])*np.linalg.norm(ax[1])))*20)
  # Crossing must be ahead of both bindings (+X). Right tail is lifted for the grab.
  for i,b in enumerate(ski_ids):
   R=d.xmat[b].reshape(3,3)
   res.extend((R-desired[i]).ravel()*4)
   res.extend((centers[i]-np.array([.12,.14 if i==0 else -.14,-.48 if i==0 else -.30]))*4)
  res.extend((centers[1]-centers[0]-np.array([0,-.28,.18]))*6)
  p,g=grab_points(m,d);res.extend((p-g)*80)
  A=np.zeros((3,m.nv));J=np.zeros_like(A);mujoco.mj_angmomMat(m,d,A,1);mujoco.mj_jacSubtreeCom(m,d,J,1)
  inertia=A[:,3:6]-A[:,:3]@np.linalg.solve(J[:,:3],J[:,3:6])
  res.extend([inertia[0,2]*8,inertia[1,2]*8,max(0,inertia[2,2]-inertia[0,0]+.15)*4,max(0,inertia[2,2]-inertia[1,1]+.15)*4]);res.extend((q-nom)*.03)
  pen=np.zeros(m.ngeom)
  for c in d.contact[:d.ncon]:
   if c.dist<-.0003:pen[int(c.geom[1])]=max(pen[int(c.geom[1])],-.0003-c.dist)
  res.extend(pen*1000)
  return q if return_q else np.asarray(res)
 best=None
 for seed in [0,1,2,3]:
  x=np.clip(nom[variable],lo[variable],hi[variable])
  if seed:x=np.clip(x+np.random.default_rng(seed).normal(0,.12,len(x)),lo[variable],hi[variable])
  r=least_squares(evaluate,x,bounds=(lo[variable],hi[variable]),max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun):best=r
  print('seed',seed,'cost',np.linalg.norm(r.fun),flush=True)
 q=evaluate(best.x,True);p,g=grab_points(m,d);Rs=[d.xmat[b].reshape(3,3) for b in ski_ids]
 axes=[R[:,0]/np.linalg.norm(R[:2,0]) for R in Rs];angle=math.degrees(math.acos(np.clip(np.dot(axes[0][:2],axes[1][:2]),-1,1)))
 out=dict(joints=dict(zip(names,q.tolist())),qpos=d.qpos.tolist(),grab_distance_m=float(np.linalg.norm(p-g)),cross_angle_deg=angle,board_centers=[d.xpos[b].tolist() for b in ski_ids],minimum_collision_distance_m=min([float(c.dist) for c in d.contact[:d.ncon]] or [0]),grab='right palm to raised right tail; front sections cross ahead of bindings; no finger actuation')
 (OUT/'cross_grab_pose_ik.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return out
if __name__=='__main__':solve_cross()
