"""Shared official G1 model access for helicopter 720 annotations."""
from pathlib import Path
import json
import numpy as np
import mujoco
from runtime import ROOT, Terrain
OUT=ROOT.parents[1]/'docs/imgs/keyframe'

def model():
 m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/g1_tabletop_steep.xml'))
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
 return m,d

def ids(m):
 j=m.actuator_trnid[:,0]
 return m.jnt_qposadr[j],m.jnt_dofadr[j],[m.joint(int(i)).name for i in j]
