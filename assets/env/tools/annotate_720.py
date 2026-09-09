"""G1 upright 720 safety-touch reference, with explicit feasibility labels."""
from pathlib import Path
import json, math
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from runtime import ROOT,Terrain
OUT=ROOT.parents[1]/'docs/imgs/keyframe'

def model():
 m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/g1_tabletop_steep.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
 return m,d

def ids(m):
 j=m.actuator_trnid[:,0];return m.jnt_qposadr[j],m.jnt_dofadr[j], [m.joint(int(i)).name for i in j]

def palm_target(m,d):
 b=m.body('left_wrist_yaw_link').id;hand=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.10,.003,-.018])
 b=m.body('left_ski_segment_2').id;target=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([-.075,.042,.010])
 return hand,target

def grab_pose():
 m,d=model();qa,va,names=ids(m);d.qpos[:3]=[0,0,30];d.qpos[3:7]=[1,0,0,0]
 nominal=np.zeros(29)
 for side in ['left','right']:
  for j,v in [('hip_pitch_joint',-1.25),('knee_joint',2.0),('ankle_pitch_joint',-.75),('hip_roll_joint',.12 if side=='left' else -.12),('ankle_roll_joint',-.12 if side=='left' else .12)]:nominal[names.index(side+'_'+j)]=v
 for j,v in [('left_shoulder_pitch_joint',.25),('left_shoulder_roll_joint',.25),('left_shoulder_yaw_joint',0),('left_elbow_joint',.6),('right_shoulder_roll_joint',-.5),('right_elbow_joint',.4)]:nominal[names.index(j)]=v
 jids=m.actuator_trnid[:,0];lo=m.jnt_range[jids,0]+.025;hi=m.jnt_range[jids,1]-.025
 variable=[i for i,n in enumerate(names) if (n.startswith('left_') and n not in ['left_hip_yaw_joint','left_hip_roll_joint','left_ankle_roll_joint']) or n in ['waist_pitch_joint']]
 lo[names.index('waist_pitch_joint')]=-.4;hi[names.index('waist_pitch_joint')]=.25
 lo[names.index('left_hip_pitch_joint')]=-2.48;hi[names.index('left_hip_pitch_joint')]=-1.1
 lo[names.index('left_knee_joint')]=1.8;hi[names.index('left_knee_joint')]=2.85
 def residual(x):
  q=nominal.copy();q[variable]=x
  # Symmetric leg tuck keeps the boards parallel; same-side hand supplies the gesture.
  for suffix in ['hip_pitch_joint','knee_joint','ankle_pitch_joint']:
   q[names.index('right_'+suffix)]=q[names.index('left_'+suffix)]
  d.qpos[qa]=q;mujoco.mj_forward(m,d);p,t=palm_target(m,d)
  residual=list((p-t)*50)+list((q-nominal)*.04)
  residual += [4*(q[names.index('left_hip_pitch_joint')]+q[names.index('left_knee_joint')]+q[names.index('left_ankle_pitch_joint')])]
  # Fixed length penalties for nonadjacent robot/board overlap.
  penetration=np.zeros(m.ngeom)
  for c in d.contact[:d.ncon]:
   if c.dist<-.0005:penetration[int(c.geom[1])]=max(penetration[int(c.geom[1])],-.0005-c.dist)
  return np.r_[residual,penetration*1000]
 res=least_squares(residual,np.clip(nominal[variable],lo[variable],hi[variable]),bounds=(lo[variable],hi[variable]),max_nfev=600,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 residual(res.x);p,t=palm_target(m,d)
 out=dict(joints=dict(zip(names,d.qpos[qa].tolist())),palm_point_world=p.tolist(),target_world=t.tolist(),distance_m=float(np.linalg.norm(p-t)),minimum_collision_distance_m=min([float(c.dist) for c in d.contact[:d.ncon]] or [0]),qpos=d.qpos.tolist())
 (OUT/'grab_pose_ik.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
 return out
if __name__=='__main__':grab_pose()
