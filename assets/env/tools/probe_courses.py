"""Exploratory unactuated 18 kg payload-on-ski probes, NOT a controlled G1 test."""
import json
import numpy as np
from validate_assets import fixture
from runtime import Terrain
from build_assets import ROOT

def main():
    results=[]
    for name,x,v,duration in [('single_bump',10,4,2),('moguls_regular',8,4,5),('tabletop_small',18,6,2)]:
        m,d,s=fixture(name,load=18);t=Terrain(name);d.qpos[0]=x;d.qpos[2]=t.height(x,0)+.018
        n=t.normal(x,0);a=np.arctan2(n[0],n[2]);d.qpos[3:7]=[np.cos(a/2),0,np.sin(a/2),0];d.qvel[:3]=[v*np.cos(a),0,-v*np.sin(a)]
        minimum=0;air=[];recontact=0;maxforce=0;after_lip_air=[]
        for k in range(round(duration/m.opt.timestep)):
            s.step(d);cs=s.contacts(d)
            minimum=min(minimum,min([c['distance'] for c in cs] or [0]))
            maxforce=max(maxforce,sum(max(c['force_contact_frame'][0],0) for c in cs))
            if not cs:
                air.append(float(d.time))
                if d.qpos[0]>22:after_lip_air.append(float(d.qpos[0]))
            elif air: recontact+=1
        results.append(dict(course=name,initial_x=x,initial_speed=v,duration=duration,payload_mass_kg=18,payload_inertia='binding inertia unchanged; free passive load fixture, not robot body',min_contact_distance_m=minimum,peak_normal_force_N=maxforce,final_position=d.qpos[:3].tolist(),airborne_samples=len(air),after_lip_air_x_extent=[min(after_lip_air),max(after_lip_air)] if after_lip_air else [],recontact_samples=recontact,warnings=d.warning.number.tolist(),penetration_screen_pass=minimum>-.025))
    report=dict(scope='Exploratory, no active controller. Failures remain visible and do not certify G1 skills.',results=results,assessment='Loaded free-ski moguls exceed the 25 mm penetration screen. The high-slope tabletop fixture does not reach the lip, so it does not validate takeoff. Full M0 / course dynamics are not accepted. Diagnose posture, payload inertia, collision and dt before training.')
    (ROOT/'reports/course_probes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
