"""Reference stepping demo; holds joint targets, contains no balance policy or root forces."""
import argparse
import json
import time
import mujoco
from build_assets import ROOT
from runtime import SnowStepper

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',default='g1_plane_00');parser.add_argument('--seconds',type=float,default=1);parser.add_argument('--viewer',action='store_true');args=parser.parse_args()
    path=ROOT/'scenes'/f'{args.scene}.xml'
    if not path.is_file(): parser.error('Unknown scene')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    if m.nkey:mujoco.mj_resetDataKeyframe(m,d,0)
    meta='g1_contacts' if args.scene.startswith('g1_') else args.scene+'_contacts'
    meta_path=ROOT/'configs'/f'{meta}.json'
    stepper=SnowStepper(m,json.loads(meta_path.read_text())) if meta_path.exists() else None
    def step():
        if stepper:stepper.step(d)
        else:mujoco.mj_step(m,d)
    if args.viewer:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(m,d) as viewer:
            while viewer.is_running() and d.time<args.seconds:
                start=time.perf_counter();step();viewer.sync();time.sleep(max(0,m.opt.timestep-time.perf_counter()+start))
    else:
        for _ in range(round(args.seconds/m.opt.timestep)):step()
    print(json.dumps(dict(simulation_time=d.time,nq=m.nq,nv=m.nv,nu=m.nu,contacts=d.ncon,warnings=d.warning.number.tolist(),note='No balance policy; falling is expected. Do not interpret this demo as trained skiing.')))

if __name__=='__main__':main()
