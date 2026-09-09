"""Native actuated aerial shooting probe. No external root forces or grasp latch."""
import json, math
import numpy as np
import mujoco
from annotate_720 import model,ids,OUT
from cross_720 import grab_points as palm_target
from scipy.spatial.transform import Rotation

def poses(m):
 qa,va,names=ids(m);gr=json.load(open(OUT/'cross_grab_pose_ik.json'));grab=np.array([gr['joints'][n] for n in names]);launch=grab.copy();land=np.zeros(29)
 # Boards already crossed at takeoff; the right hand then closes the approach.
 for side,sign in [('left',1),('right',-1)]:
  for key,value in [('hip_pitch_joint',-.7),('knee_joint',1.3),('ankle_pitch_joint',.01),('shoulder_pitch_joint',-.25),('shoulder_roll_joint',sign*1.1),('elbow_joint',.3)]:land[names.index(side+'_'+key)]=value
 return launch,grab,land

def blend(a,b,u,duration):
 u=np.clip(u,0,1);h=u**3*(10-15*u+6*u*u);v=30*u*u*(1-u)**2/duration;acc=60*u*(1-u)*(1-2*u)/duration**2
 return a+(b-a)*h,(b-a)*v,(b-a)*acc

def reference(t,T,poses):
 a,b,c=poses
 if t<.24*T:return blend(a,b,t/(.24*T),.24*T)
 if t<.60*T:return b.copy(),np.zeros(29),np.zeros(29)
 return blend(b,c,(t-.60*T)/(.32*T),.32*T)

def flight(Lz=9,T=1.3,dt=.0005,save=True,tilt=(0,0)):
 m,d=model();m.opt.timestep=dt;m.geom_contype[m.geom('snow_surface').id]=0;m.geom_conaffinity[m.geom('snow_surface').id]=0
 qa,va,names=ids(m);uu=np.array([i for i in range(m.nv) if i not in va]);ps=poses(m)
 d.qpos[:3]=[22.8,0,4];rot=Rotation.from_euler('xyz',[*tilt,0]).as_quat();d.qpos[3:7]=rot[[3,0,1,2]];d.qpos[qa]=ps[0];d.qvel[:]=0;mujoco.mj_forward(m,d)
 J=np.zeros((3,m.nv));A=np.zeros_like(J);mujoco.mj_jacSubtreeCom(m,d,J,1);mujoco.mj_angmomMat(m,d,A,1)
 initial_v=np.array([6*math.cos(math.radians(18)),0,6*math.sin(math.radians(18))+1.2]);d.qvel[:6]=np.linalg.solve(np.r_[J[:,:6],A[:,:6]],np.r_[initial_v,[0,0,Lz]])
 M=np.zeros((m.nv,m.nv));rows=[];peakreq=np.zeros(29);peakactual=np.zeros(29);maxerr=0;clipsteps=0;mincon=0
 for k in range(round(T/dt)+1):
  t=k*dt;mujoco.mj_forward(m,d);target,vel,acc=reference(t,T,ps)
  maxerr=max(maxerr,float(np.max(abs(target-d.qpos[qa]))))
  mujoco.mj_fullM(m,M,d.qM);aa=acc
  au=np.linalg.solve(M[np.ix_(uu,uu)],d.qfrc_passive[uu]-d.qfrc_bias[uu]-M[np.ix_(uu,va)]@aa)
  tau=M[np.ix_(va,va)]@aa+M[np.ix_(va,uu)]@au+d.qfrc_bias[va]-d.qfrc_passive[va]
  ctrl=target+(tau-m.actuator_biasprm[:,2]*vel)/m.actuator_gainprm[:,0]
  required=tau+m.actuator_gainprm[:,0]*(target-d.qpos[qa])-m.actuator_biasprm[:,2]*(vel-d.qvel[va])
  clipped=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);clipsteps+=int(np.any(abs(ctrl-clipped)>1e-8));d.ctrl[:]=clipped;peakreq=np.maximum(peakreq,abs(required))
  if k%max(1,round(.005/dt))==0:
   mujoco.mj_subtreeVel(m,d);p,g=palm_target(m,d);R=d.xmat[1].reshape(3,3)
   sx=[d.xmat[m.body(side+'_ski_segment_3').id].reshape(3,3)[:,0] for side in ['left','right']];cross=math.degrees(math.acos(np.clip(sx[0]@sx[1],-1,1)))
   rows.append(dict(time=t,cross_angle_deg=cross,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),com=d.subtree_com[1].copy(),angmom=d.subtree_angmom[1].copy(),yaw=math.atan2(R[1,0],R[0,0]),up=R[2,2],grab_distance=float(np.linalg.norm(p-g))))
  mincon=min(mincon,min([float(c.dist) for c in d.contact[:d.ncon]] or [0]))
  if k<round(T/dt):mujoco.mj_step(m,d);peakactual=np.maximum(peakactual,abs(d.actuator_force))
 spin=np.degrees(np.unwrap([r['yaw'] for r in rows]));com=np.array([r['com'] for r in rows]);times=np.array([r['time'] for r in rows]);expected=com[0]+times[:,None]*initial_v;expected[:,2]-=.5*9.81*times**2
 report=dict(tilt=list(tilt),final_rpy_deg=Rotation.from_matrix(d.xmat[1].reshape(3,3)).as_euler("xyz",degrees=True).tolist(),Lz=Lz,T=T,dt=dt,spin_deg=float(spin[-1]),min_up=float(min(r['up'] for r in rows)),com_ballistic_error_m=float(np.max(np.linalg.norm(com-expected,axis=1))),angular_momentum_drift=float(np.max(np.linalg.norm(np.array([r['angmom'] for r in rows])-[0,0,Lz],axis=1))),max_joint_tracking_error_rad=maxerr,control_clipped_fraction=clipsteps/(round(T/dt)+1),requested_peak_torques=peakreq.tolist(),actual_peak_torques=peakactual.tolist(),minimum_self_contact_distance=mincon,grab_distance_mid_m=float(rows[len(rows)//2]['grab_distance']),warnings=d.warning.number.tolist())
 if save:
  np.savez_compressed(OUT/'aerial_shooting_probe.npz',**{key:np.array([r[key] for r in rows]) for key in rows[0]},spin_unwrapped_deg=spin)
  (OUT/'aerial_shooting_probe.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True);return rows,report
if __name__=='__main__':flight()
