"""Independent acceptance audit. Does not mutate production MJCF/physics values.

Tilt fixture fixes roll/pitch with ideal no-work constraints and leaves XYZ/yaw
free. A fixed-roll test is not an unassisted robot demonstration.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from runtime import ROOT, SnowStepper, Terrain

OUT=ROOT/'reports/dynamics_audit'
PHYS=json.loads((ROOT/'configs/design.json').read_text())['physics']
REPORT={'engine':mujoco.__version__,'platform':platform.platform(),'scope':'Frozen production assets; independent no-work orientation fixture and explicitly labelled free-body probes. No production physics retuning.','thresholds':{'static_force_relative':.03,'static_sink_m':.005,'drop_sink_m':.025,'velocity_dt_relative':.05,'impact_impulse_dt_relative':.10,'energy_gain_fraction':.02},'results':{}}

def xml_assets_absolute(root):
    for item in root.find('asset'):
        if 'file' in item.attrib:item.set('file',str((ROOT/'scenes'/item.get('file')).resolve()))

def make(roll=0,pitch=0,yaw=0,load=18,dt=.001,free=False,course=None,flat_hfield=False,payload_inertia=False):
    root=ET.parse(ROOT/'scenes/ski_single.xml').getroot();xml_assets_absolute(root)
    ski=root.find("worldbody/body[@name='ski0_mount']");world=root.find('worldbody');world.remove(ski)
    ski.set('pos','0 0 0');ski.remove(ski.find('freejoint'))
    inert=ski.find('inertial');inert.set('mass',str(.25+load))
    if payload_inertia:
        old=np.fromstring(inert.get('diaginertia'),sep=' ');old+=load*np.array([.2**2+.1**2,.3**2+.1**2,.3**2+.2**2])/12;inert.set('diaginertia',' '.join(map(str,old)))
    carrier=ET.SubElement(world,'body',name='carrier',pos='0 0 0')
    ET.SubElement(carrier,'inertial',pos='0 0 0',mass='.001',diaginertia='.000001 .000001 .000001')
    if free:ET.SubElement(carrier,'freejoint',name='root')
    else:
        for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']):ET.SubElement(carrier,'joint',name=['x','y','z'][i],type='slide',axis=axis,damping='0')
        ET.SubElement(carrier,'joint',name='yaw',type='hinge',axis='0 0 1',damping='0')
    # yaw is a world-Z generalized coordinate; tilt is fixed beneath it.
    tilt=ET.SubElement(carrier,'body',name='tilt',euler=f'{math.radians(roll)} {math.radians(pitch)} 0')
    tilt.append(ski)
    if course:
        terrainroot=ET.parse(ROOT/'scenes'/f'{course}.xml').getroot();xml_assets_absolute(terrainroot)
        for h in terrainroot.find('asset').findall('hfield'):root.find('asset').append(copy.deepcopy(h))
        snow=world.find("geom[@name='snow_surface']");world.remove(snow)
        world.append(copy.deepcopy(terrainroot.find("worldbody/geom[@name='snow_surface']")))
    if flat_hfield:
        h=ET.SubElement(root.find('asset'),'hfield',name='audit_flat',nrow='33',ncol='321',size='20 2 .01 .5')
        snow=world.find("geom[@name='snow_surface']");snow.set('type','hfield');snow.set('hfield','audit_flat');snow.attrib.pop('size',None)
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));m.opt.timestep=dt
    d=mujoco.MjData(m)
    if free:d.qpos[2]=.04
    else:d.qpos[2]=.04;d.qpos[3]=math.radians(yaw)
    s=SnowStepper(m,json.loads((ROOT/'configs/ski_single_contacts.json').read_text()))
    return m,d,s

def snapshot(m,d,s):
    total=np.zeros(3);loads={'base':0.,'edge_L':0.,'edge_R':0.};min_dist=0.;pointloads=[]
    for i in range(d.ncon):
        c=d.contact[i];g1,g2=map(int,c.geom)
        if s.snow not in [g1,g2]:continue
        gid=g2 if g1==s.snow else g1
        if gid not in s.items:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        sign=1 if g1==s.snow else -1;wf=sign*c.frame.reshape(3,3).T@f[:3];total+=wf
        item=s.items[gid];key='base' if item['kind']=='base' else 'edge_L' if item['side']==1 else 'edge_R'
        loads[key]+=max(f[0],0);min_dist=min(min_dist,float(c.dist))
        if f[0]>1e-4:pointloads.append((float(c.pos[0]),float(c.pos[1]),float(f[0]),key,float(c.friction[1])))
    return total,loads,min_dist,pointloads

def run(m,d,s,seconds,force=None,trace=None):
    peak=0.;sink=0.;impulse=np.zeros(3);work=0.;max_energy=-np.inf;energies=[];no_contact=0;peak_state=None
    rows=[];start=time.perf_counter();cid=m.body('carrier').id
    for k in range(round(seconds/m.opt.timestep)):
        if force is not None:d.xfrc_applied[cid,:3]=force
        s.prepare(d)
        # Read consistent qpos/qvel/kinematics BEFORE integration.
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);en=float(d.energy.sum())
        max_energy=max(max_energy,en);energies.append(en)
        mujoco.mj_step2(m,d)
        f,loads,dist,pts=snapshot(m,d,s);sink=min(sink,dist);impulse+=f*m.opt.timestep
        if not pts:no_contact+=1
        if np.linalg.norm(f)>peak:
            peak=float(np.linalg.norm(f));peak_state={'time':float(d.time),'force':f.tolist(),'position':d.qpos[:3].tolist(),'distance':dist,'loads':loads,'qvel_norm':float(np.linalg.norm(d.qvel))}
        if force is not None:work+=float(np.dot(force,d.qvel[:3]))*m.opt.timestep
        if k%10==0:rows.append([float(d.time),*d.qpos[:3].tolist(),*d.qvel[:3].tolist(),*f.tolist(),dist,en,*loads.values()])
    if trace:
        np.savetxt(OUT/f'{trace}.csv',np.asarray(rows),delimiter=',',header='time,x,y,z,vx,vy,vz,Fx,Fy,Fz,min_contact_distance,mechanical_energy,base_load,left_edge_load,right_edge_load',comments='')
    return {'peak_force_N':peak,'minimum_contact_distance_m':sink,'impulse_world_Ns':impulse.tolist(),'applied_work_J':work,'initial_energy_J':energies[0],'final_energy_J':energies[-1],'maximum_energy_J':max_energy,'no_contact_steps':no_contact,'peak_state':peak_state,'final_qpos':d.qpos.tolist(),'final_qvel':d.qvel.tolist(),'warnings':d.warning.number.tolist(),'steps_per_second_including_diagnostics':round(seconds/m.opt.timestep/(time.perf_counter()-start))}

def static_and_grip():
    rows=[]
    for angle in [0,5,10,20,35,-20,-35]:
        m,d,s=make(roll=angle);r=run(m,d,s,1)
        s.prepare(d);mujoco.mj_fwdActuation(m,d);mujoco.mj_fwdAcceleration(m,d);mujoco.mj_fwdConstraint(m,d)
        f,loads,dist,pts=snapshot(m,d,s);weight=float(m.body_mass.sum()*9.81)
        rows.append({'roll_deg':angle,'weight_N':weight,'support_N':float(f[2]),'force_relative_error':abs(f[2]-weight)/weight,'loads':loads,'minimum_distance_m':dist,'active_point_count':len(pts),'mu_lateral_effective_capacity':sum(p[2]*p[4] for p in pts)/sum(p[2] for p in pts),'longitudinal_support_span_m':max(p[0] for p in pts)-min(p[0] for p in pts),'flex_qpos':d.qpos[4:].tolist()})
    REPORT['results']['static_roll_load_distribution']=rows
    rows=[]
    for roll,fy in [(0,5),(0,25),(35,25),(35,100),(35,150),(-35,25)]:
        m,d,s=make(roll=roll);run(m,d,s,.8);y0=float(d.qpos[1]);r=run(m,d,s,.4,force=np.array([0.,fy,0.]));rows.append({'roll_deg':roll,'lateral_external_force_N':fy,'lateral_displacement_m':float(d.qpos[1]-y0),'final_lateral_velocity':float(d.qvel[1]),'external_work_J':r['applied_work_J'],'max_force_N':r['peak_force_N']})
    REPORT['results']['static_friction_and_breakaway']=rows

def carve():
    rows=[]
    for roll in [0,20,-20,35,-35]:
        for dt in ([.001,.0005] if abs(roll)==35 else [.001]):
            m,d,s=make(roll=roll,dt=dt);run(m,d,s,.8);d.qvel[0]=4.;start=d.qpos[:4].copy()
            r=run(m,d,s,2,trace=f'carve_{roll}_{dt}');vx,vy=d.qvel[:2];heading=math.atan2(vy,vx);yaw=float(d.qpos[3]);distance=np.linalg.norm(d.qpos[:2]-start[:2]);r.update(roll_deg=roll,dt=dt,heading_change_deg=math.degrees(heading),yaw_change_deg=math.degrees(yaw-start[3]),sideslip_deg=math.degrees(math.atan2(math.sin(heading-yaw),math.cos(heading-yaw))),estimated_radius_m=float(distance/abs(heading)) if abs(heading)>1e-6 else None,external_yaw_torque=0,fixture='fixed roll/pitch, XYZ and yaw free, no work by fixed orientation constraints')
            rows.append(r)
    REPORT['results']['free_yaw_carving_probe']=rows

def plane_hfield():
    rows=[]
    for hfield in [False,True]:
        for roll in [0,35]:
            m,d,s=make(roll=roll,flat_hfield=hfield);run(m,d,s,.8);d.qvel[0]=4;r=run(m,d,s,1,trace=f'flat_hfield_{hfield}_roll_{roll}');r.update(surface='flat_hfield' if hfield else 'plane',roll_deg=roll);rows.append(r)
    REPORT['results']['plane_vs_flat_heightfield']=rows

def drop():
    rows=[]
    for hfield in [False,True]:
        for dt in [.001,.0005,.00025]:
            m,d,s=make(dt=dt,flat_hfield=hfield);d.qpos[2]=.30
            # Capture first impact only, ending 80ms after first contact rather than
            # letting static support make the impulse-convergence test vacuous.
            first=None;imp=np.zeros(3);peak=0;minimum=0;before=None;end=None
            for k in range(round(1/dt)):
                vpre=d.qvel[:3].copy();s.step(d);f,_,dist,pts=snapshot(m,d,s)
                if first is None and pts:first=d.time;before=vpre.copy()
                if first is not None:
                    imp+=f*dt;peak=max(peak,float(f[2]));minimum=min(minimum,dist)
                    if d.time>=first+.08:end=d.qvel[:3].copy();break
            rows.append({'surface':'flat_hfield' if hfield else 'plane','dt':dt,'first_contact_time':first,'impact_window_s':.08,'impact_impulse_Ns':imp.tolist(),'peak_force_N':peak,'min_distance_m':minimum,'velocity_before':before.tolist() if before is not None else None,'velocity_after':end.tolist() if end is not None else None})
    REPORT['results']['impact_window_dt_convergence']=rows

def terrain_failures():
    rows=[]
    for course in ['moguls_regular','tabletop_small']:
        for realistic in [False,True]:
            for dt in [.001,.0005]:
                m,d,s=make(free=True,course=course,dt=dt,payload_inertia=realistic);t=Terrain(course);x=8 if course=='moguls_regular' else 18;v=4 if x==8 else 6
                d.qpos[0]=x;d.qpos[2]=t.height(x,0)+.018;n=t.normal(x,0);a=math.atan2(n[0],n[2]);d.qpos[3:7]=[math.cos(a/2),0,math.sin(a/2),0];d.qvel[:3]=[v*math.cos(a),0,-v*math.sin(a)]
                r=run(m,d,s,5 if x==8 else 2,trace=f'{course}_physical_inertia_{realistic}_{dt}');r.update(course=course,dt=dt,payload_inertia='18 kg uniform 0.30x0.20x0.10 m box added at binding COM' if realistic else 'legacy 18 kg mass with unchanged binding inertia',payload_is_g1=False);rows.append(r)
    REPORT['results']['high_terrain_failure_isolation']=rows

def structural():
    checks=[]
    for path in sorted((ROOT/'scenes').glob('*.xml')):
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
        if m.nkey:mujoco.mj_resetDataKeyframe(m,d,0)
        mujoco.mj_forward(m,d)
        checks.append({'scene':path.name,'nq':m.nq,'nv':m.nv,'nu':m.nu,'initial_min_contact_distance':min([float(c.dist) for c in d.contact[:d.ncon]] or [0]),'equality_constraints':m.neq,'positive_mass_inertia':bool(np.all(m.body_mass[1:]>0) and np.all(m.body_inertia[1:]>0))})
    REPORT['results']['scene_structure']=checks

def save():
    REPORT['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['scenes','meshes','terrain','configs'] for p in sorted((ROOT/directory).glob('*')) if p.is_file()}
    REPORT['production_assets_modified']=False;REPORT['training_ready']=False
    (OUT/'audit.json').write_text(json.dumps(REPORT,indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,fn in [('structure',structural),('static and grip',static_and_grip),('free-yaw carving',carve),('plane vs heightfield',plane_hfield),('impact convergence',drop),('high terrain failures',terrain_failures)]:
        fn();save();print('Completed '+name,flush=True)
    print(OUT/'audit.json',flush=True)

if __name__=='__main__':main()
