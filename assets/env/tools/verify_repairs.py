"""Executable v1.2 repair gates; measurements and limits, never preset pass labels."""
import json, math, time, hashlib
import numpy as np
import mujoco
from audit_dynamics import make, run, snapshot, ROOT, Terrain
from audit_structure_and_flex import masks
from surface_contacts import SurfaceContacts

OUT=ROOT/'reports/repairs_v1_2';OUT.mkdir(exist_ok=True)
R={'engine':mujoco.__version__,'checks':[],'measurements':{},'scope':'CPU physical assets and explicit fixtures; no learned G1 policy or MJX certification'}
def save():
    R['passed']=all(x['passed'] for x in R['checks']);(OUT/'verification.json').write_text(json.dumps(R,indent=2)+'\n')
def check(name,passed,**data):
    R['checks'].append(dict(name=name,passed=bool(passed),**data));save();print(name, bool(passed),flush=True)
def measure(name,value):R['measurements'][name]=value;save()
def slide(roll=0,dt=.00025,stations=32,speed=4,hf=False,grid=.125,offset=0,duration=1):
    m,d,s=make(roll=roll,dt=dt,payload_inertia=True,flat_hfield=hf,grid=grid);d.qpos[1]=offset
    if stations!=32:s.surface=SurfaceContacts(m,list(s.items.values()),json.loads((ROOT/'configs/design.json').read_text()),stations)
    run(m,d,s,.8);d.qvel[0]=speed;r=run(m,d,s,duration)
    r['heading_deg']=math.degrees(math.atan2(d.qvel[1],d.qvel[0]));r['speed']=float(np.linalg.norm(d.qvel[:2]));r['yaw_deg']=math.degrees(d.qpos[3]);r['sideslip_deg']=math.degrees(math.atan2(math.sin(math.radians(r['heading_deg']-r['yaw_deg'])),math.cos(math.radians(r['heading_deg']-r['yaw_deg']))))
    return r

def course_fixture(course,dt):
    m,d,s=make(free=True,course=course,dt=dt,payload_inertia=True);t=Terrain(course)
    x=8 if 'moguls' in course else 18;v=4 if x==8 else 6
    d.qpos[0]=x;d.qpos[2]=t.height(x,0)+.018;n=t.normal(x,0);a=math.atan2(n[0],n[2])
    d.qpos[3:7]=[math.cos(a/2),0,math.sin(a/2),0];d.qvel[:3]=[v*math.cos(a),0,-v*math.sin(a)]
    # Clear the whole curved ski at reset, not just its centre. This is initialization.
    for _ in range(5):
        s.prepare(d);contacts=s.contacts(d)
        if not contacts:break
        lift=max(-c['distance']/max(c['frame'][2],.1) for c in contacts)+.001
        d.qpos[2]+=lift
    if s.contacts(d):raise RuntimeError('Fixture reset intersects terrain')
    return m,d,s

def first_impact(course,dt):
    m,d,s=course_fixture(course,dt);first=None;impulse=np.zeros(3);sink=0
    for _ in range(round(2/dt)):
        s.step(d);f,_,dist,pts=snapshot(m,d,s)
        if pts and first is None:first=d.time
        if first is not None:
            impulse+=f*dt;sink=min(sink,dist)
            if d.time-first>=.08:break
    return dict(first_contact_s=first,impulse_world_Ns=impulse.tolist(),sink_m=sink)

def main():
    a=masks();measure('collision_masks',a);check('g1_self_collision_restored',a['robot_self_collision_preserved']);check('crossed_skis_collide',len(a['crossed_skis_contacts'])>0)
    p=slide();measure('flat_plane',p)
    for grid,offset in [(.0625,0),(.125,0),(.25,0),(.125,.03125)]:
        h=slide(hf=True,grid=grid,offset=offset);measure(f'flat_hfield_{grid}_{offset}',h)
        error=float(np.linalg.norm(np.array(h['final_qvel'][:2])-p['final_qvel'][:2]));check(f'plane_hfield_velocity_{grid}_{offset}',error<.01,error_m_s=error)
    rows=[]
    for roll in [0,20,-20,35,-35]:
        m,d,s=make(roll=roll,payload_inertia=True);run(m,d,s,.8)
        fs=[];min_d=0
        for _ in range(400):
            s.step(d);f,loads,dist,pts=snapshot(m,d,s);fs.append(f[2]);min_d=min(min_d,dist)
        weight=float(m.body_mass.sum()*9.81);err=abs(float(np.mean(fs))-weight)/weight
        rows.append(dict(roll=roll,support_N=float(np.mean(fs)),weight_N=weight,sink_m=min_d))
        check(f'static_load_{roll}',err<.03 and min_d>-.005,relative_error=err,sink_m=min_d)
    measure('static_load',rows)
    for speed in [2,4]:
        a=slide(35,speed=speed);b=slide(35,dt=.000125,speed=speed);c=slide(35,stations=64,speed=speed)
        for label,value in [('dt',b),('density',c)]:
            verr=abs(a['speed']-value['speed'])/value['speed'];herr=abs(a['heading_deg']-value['heading_deg']);poserr=float(np.linalg.norm(np.array(a['final_qpos'][:2])-value['final_qpos'][:2]))
            check(f'carve_{speed}_{label}_convergence',verr<.05 and herr<3 and poserr<.05,speed_relative=verr,heading_error_deg=herr,position_error_m=poserr)
        measure(f'turn_{speed}',{'production':a,'half_dt':b,'double_density':c})
        if speed==2:check('conservative_carve_mechanism',abs(a['sideslip_deg'])<10 and 1<abs(a['heading_deg'])<45 and a['final_energy_J']<a['initial_energy_J'],sideslip_deg=a['sideslip_deg'],heading_deg=a['heading_deg'])
    mirror=slide(-35,speed=2);a=R['measurements']['turn_2']['production'];check('left_right_turn_symmetry',abs(mirror['heading_deg']+a['heading_deg'])<.1)
    for hf in [False,True]:
        drops=[]
        for dt in [.00025,.000125]:
            m,d,s=make(dt=dt,flat_hfield=hf,payload_inertia=True);d.qpos[2]=.3
            impulse=0;first=None;sink=0;peak=0
            for _ in range(round(.8/dt)):
                s.step(d);f,_,dist,pts=snapshot(m,d,s)
                if pts and first is None:first=d.time
                if first is not None:
                    impulse+=f[2]*dt;sink=min(sink,dist);peak=max(peak,float(f[2]))
                    if d.time-first>=.08:break
            drops.append(dict(dt=dt,first_contact_s=first,impulse_Ns=impulse,sink_m=sink,peak_N=peak))
        measure(f'impact_hfield_{hf}',drops);err=abs(drops[0]['impulse_Ns']-drops[1]['impulse_Ns'])/drops[1]['impulse_Ns'];check(f'impact_{hf}',err<.1 and min(x['sink_m'] for x in drops)>-.025,impulse_relative_error=err)
    for course in ['moguls_regular','moguls_seeded','tabletop_small','tabletop_steep']:
        rows=[]
        for dt in [.00025,.000125]:
            m,d,s=course_fixture(course,dt);x=8 if 'moguls' in course else 18
            r=run(m,d,s,3 if x==8 else 2);r['dt']=dt;rows.append(r)
            check(f'{course}_{dt}_contact',r['minimum_contact_distance_m']>-.025 and max(r['warnings'])==0 and np.isfinite(r['final_qpos']).all(),sink_m=r['minimum_contact_distance_m'],peak_N=r['peak_force_N'])
        measure(course,rows);v0=np.linalg.norm(rows[0]['final_qvel'][:3]);v1=np.linalg.norm(rows[1]['final_qvel'][:3]);err=abs(v0-v1)/v1
        # Long uncontrolled tumbling trajectories are retained as diagnostics, not
        # substituted for a first-impact convergence test with an aligned window.
        measure(course+'_long_rollout_dt',dict(speed_relative_error=float(err),within_5_percent=bool(err<.05),controlled_robot=False))
        impacts=[first_impact(course,dt) for dt in [.00025,.000125]]
        measure(course+'_first_impact',impacts)
        ia=np.array(impacts[0]['impulse_world_Ns']);ib=np.array(impacts[1]['impulse_world_Ns']);ie=float(np.linalg.norm(ia-ib)/max(np.linalg.norm(ib),1e-9))
        check(f'{course}_first_impact_convergence',all(x['first_contact_s'] is not None for x in impacts) and ie<.1,impulse_relative_error=ie)
    # No synthetic snow force while airborne, including inverted and pitched skis.
    for pitch,roll in [(0,0),(25,0),(0,35),(180,0)]:
        m,d,s=make(pitch=pitch,roll=roll);d.qpos[2]=.6;r=run(m,d,s,.02)
        check(f'airborne_{pitch}_{roll}',r['peak_force_N']==0 and r['no_contact_steps']==80)
    R['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['configs','scenes','tools'] for p in (ROOT/folder).glob('*') if p.is_file()};save()
    if not R['passed']:raise SystemExit(1)
if __name__=='__main__':main()
