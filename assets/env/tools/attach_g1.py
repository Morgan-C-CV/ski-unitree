"""Compose official 29-DoF G1 with passive skis; preserve all active joints and limits."""
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from build_assets import ROOT, ski_assets, ski_body, save_xml, fmt
from runtime import Terrain

def main():
    source=ROOT/'robots/unitree_g1/g1.xml'
    original=ET.parse(source).getroot()
    src=mujoco.MjModel.from_xml_path(str(source))
    sd=mujoco.MjData(src)
    if src.nkey: mujoco.mj_resetDataKeyframe(src,sd,0)
    names=[src.joint(int(src.actuator_trnid[i,0])).name for i in range(src.nu)]
    report=dict(source=json.loads((source.parent/'source.json').read_text()),active_joint_order=names,robot_mass_kg=float(src.body_mass.sum()),actuators=[],installation={},scenes=[])
    for i,name in enumerate(names):
        j=src.joint(name).id
        report['actuators'].append(dict(name=name,range_rad=src.jnt_range[j].tolist(),joint_force_range_Nm=src.jnt_actfrcrange[j].tolist(),kp=float(src.actuator_gainprm[i,0]),kv=float(-src.actuator_biasprm[i,2])))
    for course in json.loads((ROOT/'manifest.json').read_text())['courses']:
        name=course['name']; root=ET.parse(ROOT/'scenes'/f'{name}.xml').getroot()
        default=copy.deepcopy(original.find('default'))
        # dampratio is compiler-derived from inertia. Freeze the source compiled kv
        # so mounting passive masses does not silently retune the 29 actuators.
        default.find(".//default[@class='g1']/position").attrib.pop('dampratio',None)
        default.find(".//default[@class='collision']/geom").set('contype','4')
        # Bits: snow=1, ski=2, robot=4. Include robot bit to preserve self-collision.
        default.find(".//default[@class='collision']/geom").set('conaffinity','7')
        # Installed feet must not also directly support against snow through the blade.
        foot=default.find(".//default[@class='foot']/geom")
        foot.set('contype','0');foot.set('conaffinity','0')
        root.append(default)
        asset=root.find('asset')
        for elem in original.find('asset'):
            elem=copy.deepcopy(elem)
            if elem.tag=='mesh': elem.set('file','../robots/unitree_g1/assets/'+elem.get('file'))
            asset.append(elem)
        ski_assets(asset)
        robot=copy.deepcopy(original.find("worldbody/body[@name='pelvis']"))
        metadata=[]
        for side in ['left','right']:
            body=robot.find(f".//body[@name='{side}_ankle_roll_link']")
            # Original spherical sole bottom is ankle z=-0.035. Binding top is +0.045.
            _,items=ski_body(body,side+'_ski',(.035,0,-.080),False)
            # Do not inherit g1 joint damping/friction/armature on passive ski joints.
            metadata+=items
            report['installation'][side]=dict(parent=side+'_ankle_roll_link',translation_m=[.035,0,-.080],rotation='identity; ankle +X agrees with ski +X',binding_top_z_in_ankle=-.035)
        root.find('worldbody').append(robot)
        for tag in ['actuator','sensor']:
            if original.find(tag) is not None:
                element=copy.deepcopy(original.find(tag))
                if tag=='actuator':
                    for i,actuator in enumerate(element): actuator.set('kv',str(float(-src.actuator_biasprm[i,2])))
                root.append(element)
        path=ROOT/'scenes'/f'g1_{name}.xml';save_xml(root,path)
        m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
        for jname in names: d.qpos[m.jnt_qposadr[m.joint(jname).id]]=sd.qpos[src.jnt_qposadr[src.joint(jname).id]]
        d.qpos[0]=2;d.qpos[1]=0
        mujoco.mj_forward(m,d);t=Terrain(name)
        clearance=[]
        for item in metadata:
            g=m.geom(item['geom']).id; R=d.geom_xmat[g].reshape(3,3); center=d.geom_xpos[g]
            if item['kind']=='edge':
                for sign in [-1,1]:
                    pt=center+sign*m.geom_size[g,1]*R[:,2];clearance.append(float(t.height(*pt[:2]))-pt[2]+m.geom_size[g,0])
            else:
                mid=m.geom_dataid[g];vs=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                pts=vs@R.T+center; clearance.extend((t.height(pts[:,0],pts[:,1])-pts[:,2]).tolist())
        d.qpos[2]+=max(clearance)+.008
        # Make XML's default state and reset keyframe both valid installations.
        robot.set('pos',fmt(d.qpos[:3]))
        key=ET.SubElement(root,'keyframe');ET.SubElement(key,'key',name='ski_start',qpos=fmt(d.qpos),ctrl=fmt([d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in names]))
        save_xml(root,path)
        report['scenes'].append(dict(name=path.name,nq=m.nq,nv=m.nv,nu=m.nu,total_mass_kg=float(m.body_mass.sum())))
    (ROOT/'configs/g1_contacts.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (ROOT/'reports/g1_installation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'G1: {src.nu} unchanged actuators + 12 passive joints; {len(report["scenes"])} assembled scenes')

if __name__=='__main__': main()
