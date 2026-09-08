"""Executable asset acceptance, deterministic physics probes, and honest M0 boundary."""
import copy
import hashlib
import json
import math
import platform
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from build_assets import ROOT, CFG, design_height, ski_assets, ski_body, save_xml, fmt
from runtime import Terrain, SnowStepper

report={'engine':mujoco.__version__,'platform':platform.platform(),'checks':[],'terrain':{},'dynamics':{}}
def check(name,ok,**details):
    report['checks'].append(dict(name=name,passed=bool(ok),**details))

def fixture(course='plane_00',load=0):
    root=ET.parse(ROOT/'scenes'/f'{course}.xml').getroot();ski_assets(root.find('asset'))
    body,meta=ski_body(root.find('worldbody'),'probe',(2,0,.12),True)
    if load:
        inertial=body.find('inertial');inertial.set('mass',str(CFG['ski']['binding_mass']+load))
    # Resolve assets relative to the scenes directory while compiling in memory.
    for a in root.find('asset'):
        if 'file' in a.attrib: a.set('file',str((ROOT/'scenes'/a.get('file')).resolve()))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    d.qpos[2]+=float(Terrain(course).height(2,0))
    return m,d,SnowStepper(m,meta)

def energy(m,d):
    mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
    return float(d.energy.sum())

def run(m,d,s,seconds):
    minimum=0;peak=0;impulse=0;start=time.perf_counter();maxcon=0
    for _ in range(round(seconds/m.opt.timestep)):
        s.step(d);maxcon=max(maxcon,d.ncon)
        force=0
        for c in s.contacts(d):
            minimum=min(minimum,c['distance']);force+=max(c['force_contact_frame'][0],0)
        peak=max(peak,force);impulse+=force*m.opt.timestep
    return dict(min_contact_distance=minimum,peak_normal_force=peak,normal_impulse=impulse,max_contacts=maxcon,steps_per_second=round(seconds/m.opt.timestep/(time.perf_counter()-start)),warnings=d.warning.number.tolist())

def terrain_checks():
    rng=np.random.default_rng(41)
    for c in json.loads((ROOT/'manifest.json').read_text())['courses']:
        name=c['name']; t=Terrain(name);m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes'/f'{name}.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        # mj_ray clips a plane to its rendered size even though plane contacts are infinite.
        x=rng.uniform(.01,min(30,c['length']-.01) if c['kind']=='plane' else c['length']-.01,1500);y=rng.uniform(-c['width']/2+.05,c['width']/2-.05,1500)
        z,n,valid=t.sample(x,y); rayerr=[];raynormal=[]
        for k in range(250):
            hit=np.array([-1],np.int32);dist=mujoco.mj_ray(m,d,np.array([x[k],y[k],30.]),np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],np.uint8),1,-1,hit)
            rayerr.append(abs(30-dist-z[k]))
        check(name+'_ray_query',max(rayerr)<CFG['acceptance']['ray_height_error_m'],max_error_m=max(rayerr))
        stats=dict(triangles=c['triangles'],ray_height_error_m=max(rayerr))
        if c['kind']!='plane':
            exact=design_height(c,x,y);eps=1e-5
            gx=(design_height(c,x+eps,y)-design_height(c,x-eps,y))/(2*eps);gy=(design_height(c,x,y+eps)-design_height(c,x,y-eps))/(2*eps)
            an=np.stack([-gx,-gy,np.ones_like(gx)],-1);an/=np.linalg.norm(an,axis=-1,keepdims=True)
            angles=np.rad2deg(np.arccos(np.clip(np.sum(an*n,axis=-1),-1,1)))
            err=float(np.max(np.abs(z-exact)));normalerr=float(angles.max())
            check(name+'_resolution',err<CFG['acceptance']['terrain_height_error_m'] and normalerr<CFG['acceptance']['terrain_normal_error_deg'],height_error_m=err,normal_error_deg=normalerr)
            stats.update(height_error_m=err,normal_error_deg=normalerr,max_sampled_slope_deg=float(np.rad2deg(np.arccos(n[:,2])).max()))
        check(name+'_bounds',not bool(t.sample(-1,0)[2]) and np.isnan(t.height(-1,0)))
        check(name+'_no_duplicate_support',sum(int(m.geom_contype[g])!=0 for g in range(m.ngeom))==1)
        check(name+'_slope_group',c['slope_group_range_deg'][0]<=c['angle']<=c['slope_group_range_deg'][1])
        if c['kind'] in ['single','row','moguls','random','jump']:
            check(name+'_high_group_only',c['slope_group']=='high' and c['angle']>=20)
        if 'slope_profile' in c:
            active=[angle for x,angle in c['slope_profile'] if x<c['length']-18]
            check(name+'_variable_slope',max(active)-min(active)>=2)
            check(name+'_base_profile_in_group',all(c['slope_group_range_deg'][0]<=a<=c['slope_group_range_deg'][1] for a in active))
        report['terrain'][name]=stats
    t=Terrain('carving_gates');g=t.meta['gates'][0];x,y,_=g['center']
    check('gates_direction_and_opening',t.gate_crossing([x-1,y],[x+1,y],0) and not t.gate_crossing([x+1,y],[x-1,y],0) and not t.gate_crossing([x-1,y+5],[x+1,y+5],0))

def ballistic():
    result=[]
    for name in ['tabletop_small','tabletop_medium','tabletop_steep']:
        t=Terrain(name);a=np.deg2rad(t.meta['lip_angle_deg'])
        for v in t.meta['speed_envelope']:
            ts=np.linspace(.002,3,20000);xs=22+v*np.cos(a)*ts;zs=t.height(22,0)+v*np.sin(a)*ts-4.905*ts*ts
            ids=np.flatnonzero(zs<=t.height(xs,0));i=int(ids[0]);x=xs[i];n=t.normal(x,0);velocity=np.array([v*np.cos(a),0,v*np.sin(a)-9.81*ts[i]])
            normal_speed=-float(velocity@n);mismatch=np.rad2deg(np.arcsin(normal_speed/np.linalg.norm(velocity)))
            result.append(dict(course=name,lip_speed=v,landing_x=float(x),flight_time=float(ts[i]),region=str(t.region(x,0)),normal_approach_speed=normal_speed,tangent_mismatch_deg=float(mismatch)))
    report['ballistic']=result
    check('ballistic_lands_on_landing',all(r['region']=='landing' for r in result))

def dynamics():
    slides={}
    for yaw in [0,np.pi/2]:
        for direction in ['along','across']:
            m,d,s=fixture();d.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];run(m,d,s,.8)
            axis=np.array([np.cos(yaw),np.sin(yaw)]) if direction=='along' else np.array([-np.sin(yaw),np.cos(yaw)])
            d.qvel[:2]=axis*2; info=run(m,d,s,.5); speed=float(d.qvel[:2]@axis)
            slides[f'{yaw:.3f}_{direction}']=speed
            check(f'slide_{yaw:.3f}_{direction}',1<speed<2 and max(info['warnings'])==0,speed_after_half_second=speed)
    check('90_degree_covariance',abs(slides['0.000_along']-slides['1.571_along'])<.02 and abs(slides['0.000_across']-slides['1.571_across'])<.02)
    check('finite_anisotropic_drag',slides['0.000_along']>slides['0.000_across']+.05)
    report['dynamics']['slides']=slides
    # Unforced flight, then a real drop of a loaded flexible board; dt comparison.
    results=[]
    for dt in [.001,.0005]:
        m,d,s=fixture(load=18);m.opt.timestep=dt;d.qpos[2]=.3
        s.prepare(d);check(f'flight_zero_contacts_{dt}',d.ncon==0)
        info=run(m,d,s,1.5);info.update(dt=dt,final_z=float(d.qpos[2]),final_speed=float(np.linalg.norm(d.qvel)),flex=d.qpos[7:].tolist());results.append(info)
        check(f'loaded_drop_{dt}',info['min_contact_distance']>-.025 and max(info['warnings'])==0 and abs(d.qpos[2])<.04,**info)
    diff=abs(results[0]['normal_impulse']-results[1]['normal_impulse'])/results[1]['normal_impulse']
    check('drop_dt_impulse_convergence',diff<CFG['acceptance']['landing_impulse_relative'],relative_difference=diff)
    report['dynamics']['loaded_drop']=results
    # Gravity-only uniform slope: board points downslope with no root-force assistance.
    velocities=[]
    for dt in [.001,.0005]:
        m,d,s=fixture('plane_10');m.opt.timestep=dt;a=np.deg2rad(10);d.qpos[3:7]=[np.cos(a/2),0,np.sin(a/2),0]
        run(m,d,s,.5);v0=d.qvel[:3].copy();result=run(m,d,s,1);a_measured=float((d.qvel[:3]-v0)@np.array([np.cos(a),0,-np.sin(a)]));expected=9.81*(np.sin(a)-.05*np.cos(a));velocities.append(a_measured)
        check(f'slope_acceleration_{dt}',abs(a_measured-expected)<.15,measured=a_measured,expected=expected)
    check('slide_dt_convergence',abs(velocities[0]-velocities[1])/abs(velocities[1])<.05)
    # Static load compliance is a fixture measurement, not G1 balance evidence.
    deflections=[]
    for load in [0,9,18]:
        m,d,s=fixture(load=load);run(m,d,s,1.5);deflections.append(dict(load_kg=load,center_z=float(d.qpos[2]),flex=d.qpos[7:].tolist()))
    check('static_load_compliance',deflections[2]['center_z']<deflections[0]['center_z'],measurements=deflections)
    report['dynamics']['static_load']=deflections
    # Energy accounting includes gravity and passive springs; no control/external work.
    m,d,s=fixture();run(m,d,s,.8);d.qvel[0]=2;s.prepare(d);e0=energy(m,d);es=[]
    for _ in range(1000): s.step(d);es.append(energy(m,d))
    check('unforced_energy_dissipation',es[-1]<e0 and max(es)<e0+.05,initial=e0,final=es[-1],max=max(es))
    report['dynamics']['energy']={'initial':e0,'final':es[-1],'maximum':max(es),'applied_work':0}
    # Contact-frame orientation and smooth finite edge coefficients at prescribed roll.
    m,d,s=fixture();values=[]
    for angle in np.linspace(-.65,.65,27):
        mujoco.mj_resetData(m,d);d.qpos[2]=-.002;d.qpos[3:7]=[np.cos(angle/2),np.sin(angle/2),0,0];s.prepare(d)
        values.append(max([c.friction[1] for c in d.contact[:d.ncon]] or [0]))
        for c in d.contact[:d.ncon]:
            frame=c.frame.reshape(3,3);check('contact_frame_orthonormal',np.max(np.abs(frame@frame.T-np.eye(3)))<1e-9)
    check('edge_left_right_symmetry',np.max(np.abs(np.array(values)-values[::-1]))<1e-8,coefficients=values)
    check('edge_finite_activation',max(values)<=CFG['physics']['mu_edge']+1e-9 and min(values)>=CFG['physics']['mu_flat']-1e-9 and max(values)>.5)

def robot_checks():
    source=ROOT/'robots/unitree_g1'
    lock=json.loads((source/'source.json').read_text())
    check('official_g1_files_unchanged',all(hashlib.sha256((source/name).read_bytes()).hexdigest()==digest for name,digest in lock['sha256'].items()))
    original=mujoco.MjModel.from_xml_path(str(source/'g1.xml'))
    for p in sorted((ROOT/'scenes').glob('g1_*.xml')):
        m=mujoco.MjModel.from_xml_path(str(p));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
        joints=[m.joint(j).name for j in range(m.njnt)];passive=[j for j,n in enumerate(joints) if '_flex_' in n]
        check(p.stem+'_structure',m.nu==29 and len(passive)==12 and np.count_nonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)==1 and m.neq==0)
        check(p.stem+'_inertia',np.all(m.body_mass[1:]>0) and np.all(m.body_inertia[1:]>0))
        check(p.stem+'_initial_no_penetration',all(c.dist>-.001 for c in d.contact[:d.ncon]),ncon=d.ncon)
        check(p.stem+'_passive_unactuated',not any(j in m.actuator_trnid[:,0] for j in passive))
        check(p.stem+'_official_mass_inertia_preserved',all(np.allclose(m.body(original.body(i).name).mass,original.body_mass[i]) and np.allclose(m.body(original.body(i).name).inertia,original.body_inertia[i]) for i in range(1,original.nbody)))
        check(p.stem+'_official_actuators_preserved',np.allclose(m.actuator_gainprm,original.actuator_gainprm) and np.allclose(m.actuator_biasprm,original.actuator_biasprm) and all(np.allclose(m.jnt_actfrcrange[m.joint(original.joint(i).name).id],original.jnt_actfrcrange[i]) for i in range(1,original.njnt)))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/g1_plane_00.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);s=SnowStepper(m,json.loads((ROOT/'configs/g1_contacts.json').read_text()))
    info=run(m,d,s,.25);report['dynamics']['g1_250ms_smoke']=info
    check('g1_smoke_no_solver_warning',max(info['warnings'])==0 and np.all(np.isfinite(d.qpos)))

def main():
    terrain_checks();ballistic();dynamics();robot_checks()
    report['asset_checks_passed']=all(c['passed'] for c in report['checks'])
    report['not_yet_certified']=['carving mechanism with free yaw and controlled roll/load','dynamic edge transfer and release under full G1 loads','full G1 straight glide/turn policy','full G1 mogul and jump/landing/recovery sequences','ski segment/contact resolution convergence','MJX/batched backend compatibility','physical snow material and ski EI calibration','RL policy training and success-rate evaluation']
    report['training_ready']=False
    report['summary']='Asset and CPU probe checks only; full M0 physics gate remains open.'
    report['files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['scenes','meshes','terrain','configs','tools'] for p in sorted((ROOT/directory).glob('*')) if p.is_file()}
    (ROOT/'reports/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    failures=[c for c in report['checks'] if not c['passed']]
    print(json.dumps(dict(total=len(report['checks']),failures=failures,ballistic=report['ballistic']),indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__':main()
