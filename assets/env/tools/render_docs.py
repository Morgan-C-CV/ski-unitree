"""Render the delivered assets and static training-scene initializations to docs/imgs."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_assets import ROOT
from runtime import Terrain

OUT=ROOT.parents[1]/'docs/imgs'
FONT_PATH='/System/Library/Fonts/Supplemental/Arial.ttf'

def font(size):
    return ImageFont.truetype(FONT_PATH,size) if Path(FONT_PATH).exists() else ImageFont.load_default(size=size)

def render(scene,look,distance,azimuth,elevation,filename,label):
    m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes'/f'{scene}.xml'));d=mujoco.MjData(m)
    if m.nkey:mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.lookat[:]=look;camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
    options=mujoco.MjvOption();options.geomgroup[3]=0
    with mujoco.Renderer(m,height=840,width=1280) as renderer:
        renderer.update_scene(d,camera=camera,scene_option=options)
        # Avoid heightfield shadow-map acne; retain actual geometry and normal shading.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        frame=Image.fromarray(renderer.render())
    canvas=Image.new('RGB',(1280,910),'#102635');canvas.paste(frame,(0,0));draw=ImageDraw.Draw(canvas)
    draw.text((24,855),label,font=font(24),fill='#eef7fd')
    canvas.save(OUT/filename)
    return {'file':filename,'scene':scene,'label':label,'size':[1280,910],'state':'MJCF initial state / ski_start keyframe; no policy rollout','camera':{'lookat':list(map(float,look)),'distance':distance,'azimuth':azimuth,'elevation':elevation},'render_shadows':False}

def sheet(items,name,columns):
    w=640;h=455;im=Image.new('RGB',(w*columns,h*math.ceil(len(items)/columns)),'#102635')
    for i,item in enumerate(items):
        with Image.open(OUT/item['file']) as tile:im.paste(tile.resize((w,h),Image.Resampling.LANCZOS),((i%columns)*w,(i//columns)*h))
    im.save(OUT/name)

def main():
    OUT.mkdir(parents=True,exist_ok=True);assets=[];environments=[];training=[]
    for scene,look,dist,az,ev,label in [
        ('ski_single',[0,0,.12],1.95,110,-40,'SINGLE SKI | 7 segments / 6 passive joints'),
        ('ski_pair',[0,0,.12],2.35,115,-48,'TWIN-TIP SKIS | distributed base and edge contacts'),
        ('g1_plane_00',[2,0,.65],3.15,135,-20,'OFFICIAL G1 + SKIS | 29 actuators / 12 passive joints')]:
        assets.append(render(scene,look,dist,az,ev,'asset_'+scene+'.png',label))
    courses=json.loads((ROOT/'manifest.json').read_text())['courses']
    for c in courses:
        name=c['name'];t=Terrain(name);group=c['slope_group'].upper()
        if c['kind']=='plane':
            scene='g1_'+name;look=[2,0,float(t.height(2,0))+.5];dist=6;az=100;ev=-25
        elif c['kind']=='jump':
            scene=name;look=[24,0,float(t.height(24,0))];dist=32;az=100;ev=-42
        elif c['kind'] in ['single','row','moguls','random']:
            scene=name;cx=14 if c['kind']=='single' else 20;look=[cx,0,float(t.height(cx,0))];dist=10 if c['kind']=='single' else 31;az=135;ev=-52
        else:
            scene=name;cx=c['length']*.5;look=[cx,0,float(t.height(cx,0))];dist=c['length']*1.1;az=125;ev=-54
        label=f"{group} | {name} | nominal base {c['angle']} deg"
        environments.append(render(scene,look,dist,az,ev,'env_'+name+'.png',label))
    # Show actual robot initialization in course context. Do not stage a learned skill.
    for name in ['piste_low','carving_gates','piste_high','moguls_regular','tabletop_small','tabletop_steep']:
        t=Terrain(name);cx=9 if t.meta['kind']!='jump' else 12
        look=[cx,0,float(t.height(cx,0))+.5]
        training.append(render('g1_'+name,look,29 if cx==9 else 38,135,-47,'training_'+name+'.png',f"G1 INITIALIZATION | {name} | static scene, not trained motion"))
    sheet(assets,'assets_overview.png',3)
    sheet(environments,'environments_overview.png',4)
    sheet(training,'training_overview.png',3)
    index={'renderer':'MuJoCo '+mujoco.__version__,'source_manifest':'../../assets/env/manifest.json','note':'Native MuJoCo geometry renders. Shadow mapping disabled only for visualization. All robot poses are initial states; no trained policy or successful skill is implied.','assets':assets,'environments':environments,'training_initializations':training}
    (OUT/'render_index.json').write_text(json.dumps(index,indent=2)+'\n')
    lines=['# 资产与训练环境渲染图','', '使用 MuJoCo 3.3.7 实际渲染，未更改几何、碰撞或物理参数。为避免高度场阴影条纹，仅在渲染时关闭阴影映射，保留地形法向明暗。','', '**机器人画面均为初始化静态场景，不是训练完成后的动作或成功轨迹。**','', '## 总览','', '![资产总览](assets_overview.png)','', '![训练环境初始化](training_overview.png)','', '![全部坡度与地形](environments_overview.png)','', '## 单图索引','']
    for category,items in [('资产',assets),('环境（低／中／高坡）',environments),('G1 训练场景初始化',training)]:
        lines+=['### '+category,'']
        lines += [f"- [{item['label']}]({item['file']})" for item in items]
        lines+=['']
    lines+=['复现：在项目根目录运行 `.venv/bin/python assets/env/tools/render_docs.py`。macOS 需要图形上下文权限。完整来源、镜头和状态说明见 `render_index.json`。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Rendered {len(assets)+len(environments)+len(training)} individual PNGs and 3 overview PNGs to {OUT}')

if __name__=='__main__':main()
