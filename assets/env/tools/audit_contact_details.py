"""Diagnostic A/B probes: no production parameter modifications."""
import json
import math
import numpy as np
import mujoco
from audit_dynamics import OUT, make, run, snapshot

def normals():
    rows=[]
    for surface,grid,mode,offset in [('plane',.125,'all',0),('hfield',.125,'all',0),('hfield',.125,'all',.03125),('hfield',.0625,'all',0),('hfield',.25,'all',0),('hfield',.125,'base',0),('hfield',.125,'edge',0)]:
        m,d,s=make(flat_hfield=surface=='hfield',grid=grid,contact_mode=mode);d.qpos[1]=offset;run(m,d,s,.8);d.qvel[0]=4
        maxangle=0.;offimpulse=0.;totalimpulse=0.;worst=None;min_dist=0;maxFy=0
        for k in range(round(1/m.opt.timestep)):
            s.step(d);f,loads,dist,pts=snapshot(m,d,s);min_dist=min(min_dist,dist);maxFy=max(maxFy,abs(float(f[1])))
            for i in range(d.ncon):
                c=d.contact[i];g1,g2=map(int,c.geom)
                if s.snow not in [g1,g2]:continue
                gid=g2 if g1==s.snow else g1
                if gid not in s.items:continue
                fc=np.zeros(6);mujoco.mj_contactForce(m,d,i,fc)
                if fc[0]<=.01:continue
                n=c.frame[:3]*(1 if g1==s.snow else -1);angle=math.degrees(math.acos(np.clip(n[2],-1,1)))
                totalimpulse+=fc[0]*m.opt.timestep
                if angle>5:offimpulse+=fc[0]*m.opt.timestep
                if angle>maxangle:
                    maxangle=angle;worst={'time':float(d.time),'normal':n.tolist(),'force_contact':fc.tolist(),'position':c.pos.tolist(),'geom':m.geom(gid).name,'distance':float(c.dist)}
        rows.append({'surface':surface,'grid_m':grid,'contact_mode':mode,'y_offset_m':offset,'final_vx':float(d.qvel[0]),'final_vy':float(d.qvel[1]),'max_normal_error_deg':maxangle,'normal_impulse_fraction_more_than_5deg':offimpulse/totalimpulse,'maximum_abs_Fy_N':maxFy,'minimum_distance_m':min_dist,'worst_contact':worst})
    return rows

def transfer():
    rows=[]
    for dt in [.00025,.000125]:
        m,d,s=make(roll_motor=True,dt=dt,payload_inertia=True);d.ctrl[0]=0;run(m,d,s,.8)
        records=[];work=0.;en0=None;max_excess=-1e10;previous_force=None;max_df=0.;sink=0.;maxtracking=0
        jid=m.joint('fixture_roll').id;qadr=m.jnt_qposadr[jid];vadr=m.jnt_dofadr[jid]
        for k in range(round(2/dt)):
            t=k*dt;target=math.radians(35)*math.sin(2*math.pi*t/2);d.ctrl[0]=target
            s.prepare(d);mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d);en=float(d.energy.sum())
            if en0 is None:en0=en
            max_excess=max(max_excess,en-en0-work);q=float(d.qpos[qadr]);vel=float(d.qvel[vadr]);mujoco.mj_step2(m,d)
            work+=float(d.actuator_force[0])*vel*dt
            force,loads,dist,pts=snapshot(m,d,s);sink=min(sink,dist);maxtracking=max(maxtracking,abs(target-q))
            if previous_force is not None:max_df=max(max_df,float(np.linalg.norm(force-previous_force)))
            previous_force=force
            if k%max(1,round(.01/dt))==0:records.append([t,math.degrees(target),math.degrees(q),*force,*loads.values(),float(d.actuator_force[0]),work,en])
        np.savetxt(OUT/f'edge_transfer_{dt}.csv',records,delimiter=',',header='time,target_roll_deg,actual_roll_deg,Fx,Fy,Fz,base_load,left_load,right_load,fixture_torque_Nm,fixture_work_J,mechanical_energy_J',comments='')
        rows.append({'dt':dt,'duration':2,'applied_fixture_work_J':work,'maximum_energy_above_initial_plus_work_J':max_excess,'max_force_vector_change_per_step_N':max_df,'min_distance_m':sink,'max_tracking_error_deg':math.degrees(maxtracking),'final_position':d.qpos[:3].tolist(),'warnings':d.warning.number.tolist(),'fixture':'XYZ and yaw free; explicit bounded roll motor, torque/work recorded. Not a robot controller.'})
    return rows

def release_and_pitch():
    out=[]
    for pitch,roll in [(0,0),(25,0),(0,35),(180,0)]:
        m,d,s=make(pitch=pitch,roll=roll);d.qpos[2]=.6
        # Initial pose deliberately above ground. Follow gravity for 20ms, no contacts.
        r=run(m,d,s,.02);out.append({'pitch_deg':pitch,'roll_deg':roll,'no_contact_steps':r['no_contact_steps'],'steps':round(.02/m.opt.timestep),'force_peak':r['peak_force_N']})
    # Pure pitch relative to ground may make contacts, but must not activate grip.
    m,d,s=make(pitch=25);d.qpos[2]=-.005;s.prepare(d)
    mus=[float(c.friction[1]) for c in d.contact[:d.ncon] if s.snow in c.geom and not c.exclude]
    return {'airborne':out,'pure_pitch_max_mu_lateral':max(mus) if mus else None,'pure_pitch_contacts':len(mus)}

def main():
    OUT.mkdir(parents=True,exist_ok=True);r={}
    for key,fn in [('flat_heightfield_contact_normals',normals),('dynamic_edge_transfer',transfer),('airborne_and_pitch',release_and_pitch)]:
        r[key]=fn();(OUT/'contact_details.json').write_text(json.dumps(r,indent=2)+'\n');print('Completed '+key,flush=True)

if __name__=='__main__':main()
