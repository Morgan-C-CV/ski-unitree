"""Actual MuJoCo offscreen previews; run with desktop graphics permission on macOS."""
import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from build_assets import ROOT
from runtime import Terrain

def render(name,lookat,distance,azimuth,elevation,path):
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes'/f'{name}.xml'));d=mujoco.MjData(m)
    if m.nkey: mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.lookat[:]=lookat;camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
    options=mujoco.MjvOption();options.geomgroup[3]=0
    with mujoco.Renderer(m,height=700,width=1000) as renderer:
        renderer.update_scene(d,camera=camera,scene_option=options)
        im=Image.fromarray(renderer.render());im.save(path)
    return im

def main():
    configs=[('g1_plane_00',[2,0,.65],3.2,135,-22,'G1 + PASSIVE TWIN-TIP SKIS'),('ski_pair',[0,0,.08],2.3,115,-50,'7 SEGMENTS / 6 PASSIVE JOINTS PER SKI'),('piste_low',[48,0,0],83,145,-45,'LOW: VARIABLE 2-8 DEG / GATES'),('carving_gates',[48,0,0],85,145,-50,'MEDIUM: VARIABLE 11-19 DEG / GATES'),('piste_high',[48,0,0],94,145,-58,'HIGH: VARIABLE 22-38 DEG / GATES'),('moguls_regular',[20,0,0],27,140,-53,'HIGH ONLY: MOGULS / BASE 26-30 DEG'),('single_bump',[14,0,0],9,130,-43,'HIGH ONLY: SINGLE BUMP / BASE 22 DEG'),('tabletop_small',[24,0,0],29,100,-40,'HIGH ONLY: TABLETOP / APPROACH 22 DEG'),('tabletop_medium',[24,0,0],29,100,-43,'HIGH ONLY: TABLETOP / APPROACH 28 DEG'),('tabletop_steep',[24,0,0],30,100,-48,'HIGH ONLY: TABLETOP / APPROACH 35 DEG')]
    sheet=Image.new('RGB',(1500,1975),'#122737');draw=ImageDraw.Draw(sheet)
    for i,(name,look,distance,az,ev,label) in enumerate(configs):
        if i>=2:look[2]=float(Terrain(name).height(look[0],look[1]))
        im=render(name,look,distance,az,ev,ROOT/'reports'/f'{name}.png');im=im.resize((750,350))
        x=(i%2)*750;y=(i//2)*395;sheet.paste(im,(x,y));draw.text((x+18,y+363),label,fill='#e9f3fa')
    sheet.save(ROOT/'reports/asset_overview.png')
    print('Rendered ten MuJoCo views and asset_overview.png')

if __name__=='__main__':main()
