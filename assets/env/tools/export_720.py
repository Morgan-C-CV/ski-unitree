"""Export tip-cross helicopter targets, measured native flight, and explicit validity masks."""
import csv, hashlib, json, math, os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from annotate_720 import model, ids, OUT
from runtime import ROOT, Terrain, SnowStepper
from cross_720 import grab_points
from flight_720 import poses, blend


def main():
 m,d=model();qa,va,names=ids(m);land=poses(m)[2]
 terrain=Terrain('tabletop_steep');stepper=SnowStepper(m,json.load(open(ROOT/'configs/g1_contacts.json')));surf=stepper.surface
 air=np.load(OUT/'placed_aerial.npz');probe=json.load(open(OUT/'aerial_shooting_probe.json'));fine=json.load(open(OUT/'aerial_dt_convergence.json'));placement=json.load(open(OUT/'placement.json'));landing=json.load(open(OUT/'landing_probe.json'))
 def inspect(q):
  d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
  world=d.xpos[surf.bids]+np.einsum('nij,nj->ni',d.xmat[surf.bids].reshape(-1,3,3),surf.local)
  clearance=float(np.min(world[:,2]-surf.radii-terrain.height(world[:,0],world[:,1])))
  R=d.xmat[1].reshape(3,3);centres=[];axes=[];tips=[];tails=[]
  for side in ['left','right']:
   b=m.body(side+'_ski_segment_3').id;centres.append((d.xpos[b]-d.xpos[1])@R);axes.append(d.xmat[b].reshape(3,3)[:,0]@R)
   for out,i,x,z in [(tips,6,.2,.065),(tails,0,-.2,.05)]:
    b=m.body(side+'_ski_segment_'+str(i)).id;out.append((d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([x,0,z])).tolist())
  matrix=np.column_stack([axes[0][:2],-axes[1][:2]])
  angle=float(np.degrees(np.arccos(np.clip(np.dot(axes[0][:2],axes[1][:2])/(np.linalg.norm(axes[0][:2])*np.linalg.norm(axes[1][:2])),-1,1))))
  intersection=None;gap=None;front=False
  if abs(np.linalg.det(matrix))>.015:
   intersection=np.linalg.solve(matrix,centres[1][:2]-centres[0][:2]);gap=float(centres[1][2]+intersection[1]*axes[1][2]-centres[0][2]-intersection[0]*axes[0][2]);front=bool(np.all((intersection>.12)&(intersection<.65)));intersection=intersection.tolist()
  p,g=grab_points(m,d)
  self_dist=min([float(c.dist) for c in d.contact[:d.ncon] if stepper.snow not in c.geom and not c.exclude] or [0])
  return dict(ski_forward_intersection_m=intersection,front_sections_cross=front,cross_vertical_gap_m=gap,projected_cross_angle_deg=angle,palm_tail_distance_m=float(np.linalg.norm(p-g)),ski_tip_world=tips,ski_tail_world=tails,minimum_ski_vertical_clearance_m=clearance,minimum_self_contact_distance_m=self_dist,joint_limits_ok=bool(np.all(d.qpos[qa]>=m.jnt_range[m.actuator_trnid[:,0],0]-1e-5)&np.all(d.qpos[qa]<=m.jnt_range[m.actuator_trnid[:,0],1]+1e-5)))
 frames=[]
 def add(t,phase,q,v,ctrl,spin,dynamic):
  metrics=inspect(q);grip=(0<=t<=.78);frames.append(dict(index=len(frames),time_s=float(t),phase=phase,qpos=q.tolist(),qvel=v.tolist(),ctrl=ctrl.tolist(),body_yaw_unwrapped_deg=float(spin),metrics=metrics,targets=dict(tip_cross=bool(grip),palm_tail_touch=bool(grip),parallel_skis=bool(t>=1.2)),masks=dict(pose_reference=True,dynamic_state=bool(dynamic),velocity_reference=bool(dynamic),action_reference=False,native_controller_command_record=bool(dynamic),successful_full_skill=False,physical_grasp=False),image=f'frame_{len(frames):02d}.png'))
 # Ground phases are explicitly pose objectives. No invented velocities or actions are valid.
 # The last prelaunch target stays parallel: crossing forms at release, whose t=0 reset is already crossed.
 for t,x,knee in [(-.30,20.65,1.35),(-.15,21.3,1.7),(-.05,21.75,1.5)]:
  q=air['qpos'][0].copy();q[:3]=[x,0,0];q[3:7]=[1,0,0,0];q[7:]=0;q[qa]=land
  slope=math.atan2(float(terrain.normal(x,0)[0]),float(terrain.normal(x,0)[2]))
  for side in ['left','right']:
   ankle=float(np.clip(slope-knee*.45,-.8,.45));q[qa[names.index(side+'_knee_joint')]]=knee;q[qa[names.index(side+'_hip_pitch_joint')]]=slope-knee-ankle;q[qa[names.index(side+'_ankle_pitch_joint')]]=ankle
  q[2]+=.002-inspect(q)['minimum_ski_vertical_clearance_m'];add(t,'approach_pose_goal',q,np.zeros(m.nv),q[qa],0,False)
 for t in [0,.05,.10,.16,.22,.28,.34,.40,.46,.52,.58,.64,.70,.76,.82,.88,.94,1,1.06,1.12,1.18,1.24,1.3]:
  k=int(np.argmin(abs(air['time']-t)));phase='takeoff_air_reset' if t==0 else ('touchdown_boundary' if t==1.3 else ('cross_tail_touch' if t<=.78 else ('uncross' if t<1.2 else 'parallel_before_landing')))
  add(t,phase,air['qpos'][k],air['qvel'][k],air['ctrl'][k],air['spin_unwrapped_deg'][k],True)
 for t,knee in [(1.35,1.65),(1.45,1.8),(1.60,1.5),(1.80,1.3)]:
  q=air['qpos'][-1].copy();q[0]+=placement['initial_com_velocity'][0]*(t-1.3);q[1]=0;q[3:7]=[1,0,0,0];q[7:]=0;q[qa]=land
  slope=math.atan2(float(terrain.normal(q[0],0)[0]),float(terrain.normal(q[0],0)[2]))
  for side in ['left','right']:
   ankle=float(np.clip(slope-knee*.45,-.8,.45));q[qa[names.index(side+'_knee_joint')]]=knee;q[qa[names.index(side+'_hip_pitch_joint')]]=slope-knee-ankle;q[qa[names.index(side+'_ankle_pitch_joint')]]=ankle
  q[2]+=.002-inspect(q)['minimum_ski_vertical_clearance_m'];add(t,'landing_absorption_pose_goal',q,np.zeros(m.nv),q[qa],720,False)
 # Check every 5 ms snapshot, not only the selected rendered frames.
 measured=[inspect(q) for q in air['qpos']];hold=[v for t,v in zip(air['time'],measured) if t<=.78];parallel=[v for t,v in zip(air['time'],measured) if t>=1.2]
 limits=m.jnt_actfrcrange[m.actuator_trnid[:,0],1];checks={
  'at_least_15_frames':len(frames)>=15,
  'front_cross_at_takeoff_and_hold':all(v['front_sections_cross'] for v in hold),
  'grab_hold_palm_tail_gap_under_10mm':max(v['palm_tail_distance_m'] for v in hold)<.010,
  'parallel_before_landing_under_2deg':max(v['projected_cross_angle_deg'] for v in parallel)<2,
  'spin_within_1deg_of_720':abs(probe['spin_deg']-720)<1,
  'airborne_snow_clearance':placement['minimum_interior_vertical_clearance_m']>0,
  'native_self_penetration_under_1mm':probe['minimum_self_contact_distance']>-.001,
  'ground_goals_self_penetration_under_1mm':all(f['metrics']['minimum_self_contact_distance_m']>-.001 for f in frames),
  'ground_goals_no_snow_penetration':all(f['metrics']['minimum_ski_vertical_clearance_m']>=-1e-6 for f in frames),
  'all_keyframe_joint_limits':all(f['metrics']['joint_limits_ok'] for f in frames),
  'actual_joint_torque_limits':bool(np.all(np.array(probe['actual_peak_torques'])<=limits+1e-6)),
  'air_com_ballistic_error_under_5mm':probe['com_ballistic_error_m']<.005,
  'air_angular_momentum_relative_drift_under_1percent':probe['angular_momentum_drift']/probe['Lz']<.01,
  'dt_halving_heading_difference_under_0_2deg':abs(probe['spin_deg']-fine['spin_deg'])<.2,
  'air_no_mujoco_warnings':not any(probe['warnings']),
  'production_snow_backend_replay':json.loads((OUT/'terrain_replay_validation.json').read_text())['passed'],
 }
 report=dict(status=('AIRBORNE_REFERENCE_VALIDATED_GROUND_SKILL_NOT_VALIDATED' if all(checks.values()) else 'REFERENCE_CHECKS_FAILED'),checks=checks,airborne_checks_passed=all(checks.values()),frame_count=len(frames),native_dynamic_frames=sum(f['masks']['dynamic_state'] for f in frames),training_ready_full_skill=False,placement=placement,airborne=probe,dt_halving=dict(dt_s=fine['dt'],spin_deg=fine['spin_deg'],heading_difference_deg=abs(probe['spin_deg']-fine['spin_deg'])),hold_metrics=dict(max_palm_tail_distance_m=max(v['palm_tail_distance_m'] for v in hold),min_forward_intersection_m=np.min([v['ski_forward_intersection_m'] for v in hold],axis=0).tolist(),max_forward_intersection_m=np.max([v['ski_forward_intersection_m'] for v in hold],axis=0).tolist()),failed_fixed_pose_landing=landing,blockers=['Ground takeoff has not been dynamically generated; t=0 is an airborne curriculum reset with prescribed angular momentum.','Official G1 hands are fixed geometry; palm/tail proximity is not a physical grasp.','Fixed-pose native landing loses balance; ground frames are pose goals, not a successful dynamic demonstration.','Native right shoulder roll joint saturates at 25 Nm during release; joint tracking is not exact.'])
 (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 data=dict(schema='ski-helicopter720-front-cross-v1',phase_semantics='annotation labels, not a contact-based runtime phase estimator',model='../../../assets/env/scenes/g1_tabletop_steep.xml',units=dict(time='s',position='m',joint_angle='rad',angular_momentum='kg m^2/s',yaw_label='deg'),quaternion='MuJoCo wxyz',qpos_joints=[dict(name=m.joint(j).name,qpos_address=int(m.jnt_qposadr[j]),dof_address=int(m.jnt_dofadr[j])) for j in range(m.njnt)],actuator_joint_names=names,source_pose='reference.jpeg',controller_frequency_hz=4000,trajectory_sample_frequency_hz=200,action_note='ctrl is the native 4 kHz teacher command sampled at 200 Hz; not a validated 100 Hz LLC action target',frames=frames)
 (OUT/'keyframes.json').write_text(json.dumps(data,indent=2)+'\n')
 np.savez_compressed(OUT/'keyframes.npz',time=np.array([f['time_s'] for f in frames]),qpos=np.array([f['qpos'] for f in frames]),qvel=np.array([f['qvel'] for f in frames]),ctrl=np.array([f['ctrl'] for f in frames]),dynamic_valid=np.array([f['masks']['dynamic_state'] for f in frames]),action_valid=np.zeros(len(frames),bool),pose_valid=np.ones(len(frames),bool),physical_grasp_valid=np.zeros(len(frames),bool),full_skill_valid=np.zeros(len(frames),bool),spin_unwrapped_deg=np.array([f['body_yaw_unwrapped_deg'] for f in frames]),joint_names=np.array(names))
 with (OUT/'keyframes.csv').open('w') as file:
  writer=csv.writer(file);writer.writerow(['index','time_s','phase','yaw_unwrapped_deg','dynamic_valid','front_cross','palm_tail_distance_m',*names])
  for f in frames:writer.writerow([f['index'],f['time_s'],f['phase'],f['body_yaw_unwrapped_deg'],f['masks']['dynamic_state'],f['metrics']['front_sections_cross'],f['metrics']['palm_tail_distance_m'],*np.array(f['qpos'])[qa]])
 tree=ET.parse(ROOT/'scenes/g1_tabletop_steep.xml');root=tree.getroot()
 for element in root.iter():
  if 'file' in element.attrib:element.set('file',os.path.relpath((ROOT/'scenes'/element.get('file')).resolve(),OUT))
 old=root.find('keyframe')
 if old is not None:root.remove(old)
 keys=ET.SubElement(root,'keyframe')
 for f in frames:ET.SubElement(keys,'key',name=f"{f['index']:02d}_{f['phase']}",time=str(f['time_s']+.3),qpos=' '.join(map(str,f['qpos'])),qvel=' '.join(map(str,f['qvel'])),ctrl=' '.join(map(str,f['ctrl'])))
 ET.indent(tree);tree.write(OUT/'keyframes_scene.xml',encoding='unicode')
 loaded=mujoco.MjModel.from_xml_path(str(OUT/'keyframes_scene.xml'));assert loaded.nkey==len(frames)
 print(json.dumps({k:report[k] for k in ['status','checks','frame_count']},indent=2))

if __name__=='__main__':main()
