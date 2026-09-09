"""Exact collision-filter review and explicit-load elastic calibration fixture."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from audit_dynamics import ROOT, OUT, xml_assets_absolute

def masks():
    source=mujoco.MjModel.from_xml_path(str(ROOT/'robots/unitree_g1/g1.xml'))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/g1_plane_00.xml'))
    def knee(model,side):return next(i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name==side+'_knee_link' and model.geom_contype[i])
    pairs=[]
    for label,model in [('official',source),('ski_assembly',m)]:
        a=knee(model,'left');b=knee(model,'right')
        allowed=bool((model.geom_contype[a]&model.geom_conaffinity[b]) or (model.geom_contype[b]&model.geom_conaffinity[a]))
        pairs.append({'model':label,'pair':['left_knee_link','right_knee_link'],'masks':[[int(model.geom_contype[g]),int(model.geom_conaffinity[g])] for g in [a,b]],'pair_allowed_by_masks':allowed,'explicit_contact_pairs':model.npair})
    p=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/ski_pair.xml'));d=mujoco.MjData(p)
    right=p.jnt_qposadr[p.joint('ski1_free').id];d.qpos[right:right+3]=d.qpos[:3];d.qpos[right+3:right+7]=[np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
    mujoco.mj_forward(p,d)
    crossed=[]
    for c in d.contact[:d.ncon]:
        names=[p.geom(int(g)).name for g in c.geom]
        if any(n.startswith('ski0') for n in names) and any(n.startswith('ski1') for n in names):crossed.append({'geoms':names,'distance':float(c.dist)})
    return {'robot_self_collision':pairs,'crossed_skis_contacts':crossed,'robot_self_collision_preserved':pairs[0]['pair_allowed_by_masks']==pairs[1]['pair_allowed_by_masks']}

def beam():
    root=ET.parse(ROOT/'scenes/ski_single.xml').getroot();xml_assets_absolute(root)
    root.find('option').set('gravity','0 0 0');ski=root.find("worldbody/body[@name='ski0_mount']");ski.remove(ski.find('freejoint'))
    for geom in root.findall('.//geom'):geom.set('contype','0');geom.set('conaffinity','0')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));bid=m.body('ski0_segment_6').id
    rows=[]
    for load in [5,10,20]:
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);z0=float(d.xipos[bid,2]);d.xfrc_applied[bid,2]=-load
        for _ in range(2000):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);deflection=z0-float(d.xipos[bid,2]);expected=load*(.5**2+.3**2+.1**2)/325
        before=d.qpos.copy();d.xfrc_applied[:]=0
        for _ in range(3000):mujoco.mj_step(m,d)
        rows.append({'external_tip_load_N':load,'deflection_m':deflection,'discrete_small_angle_prediction_m':expected,'relative_error':abs(deflection-expected)/expected,'loaded_joint_qpos_rad':before.tolist(),'unloaded_max_joint_deflection_rad':float(np.max(abs(d.qpos))),'fixture':'central mount fixed, gravity/contact disabled, force at front outer segment COM x=0.6m; 3 joints at x=0.1,0.3,0.5m. This checks configured discrete stiffness, not real ski EI calibration.'})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);r={'collision_masks':masks(),'elastic_load_unload':beam()};(OUT/'structure_flex.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()
