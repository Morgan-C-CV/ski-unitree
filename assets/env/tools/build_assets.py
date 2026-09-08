"""Deterministic low-poly MuJoCo skiing assets. Run from any working directory."""
from pathlib import Path
import copy
import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'configs/design.json').read_text())

def fmt(v):
    return ' '.join(f'{float(x):.10g}' for x in np.atleast_1d(v))

def el(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k,v in kw.items()})

def save_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')

def smooth(u):
    u=np.clip(u,0,1)
    return u**3*(10-15*u+6*u*u)

def hermite(x, a, b, z0, z1, m0, m1):
    t=(x-a)/(b-a); d=b-a
    # Quintic with zero second derivative at both endpoints.
    c=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20.]]), [z1-z0-m0*d,(m1-m0)*d,0])
    return z0+m0*d*t+c[0]*t**3+c[1]*t**4+c[2]*t**5

def definitions():
    out=[]
    for angle in [0,5,10,15,20,25,30,35,40]:
        out.append(dict(name=f'plane_{angle:02d}',kind='plane',angle=angle,length=100,width=40,dx=0,dy=0))
    out += [dict(name='piste_low',kind='piste',angle=5,length=100,width=32,dx=.5,dy=2,slope_profile=[[0,2],[20,8],[40,4],[60,7],[82,3],[92,0],[100,0]]),
            dict(name='carving_gates',kind='piste',angle=15,length=100,width=32,dx=.5,dy=2,slope_profile=[[0,12],[20,18],[40,13],[60,19],[82,11],[92,0],[100,0]]),
            dict(name='piste_high',kind='piste',angle=28,length=100,width=32,dx=.5,dy=2,slope_profile=[[0,22],[20,32],[40,25],[60,38],[82,24],[92,0],[100,0]]),
            dict(name='cross_slope',kind='cross',angle=16,length=60,width=24,dx=.25,dy=.5,slope_profile=[[0,13],[16,18],[30,15],[42,12],[52,0],[60,0]])]
    for name,kind,amp,angle in [('single_bump','single',.12,22),('bump_row','row',.16,25),('moguls_regular','moguls',.2,28),('moguls_seeded','random',.2,35)]:
        out.append(dict(name=name,kind=kind,angle=angle,length=48,width=16,dx=.125,dy=.125,amplitude=amp,rx=1.4,ry=1.15,row_spacing=3.5,column_spacing=3.2,stagger=1.6,seed=CFG['seed'],slope_profile=[[0,angle],[12,angle-2],[24,angle+2],[30,angle-1],[40,0],[48,0]]))
    for name,angle,landing,lip in [('tabletop_small',22,25,12),('tabletop_medium',28,30,16),('tabletop_steep',35,35,18)]:
        out.append(dict(name=name,kind='jump',angle=angle,length=64,width=16,dx=.1,dy=2,lip_x=22,lip_angle_deg=lip,table_end=22.6,landing_angle_deg=landing,landing_end=34,runout_start=42,speed_envelope=[4,5,6],slope_profile=[[0,angle],[8,angle-2],[16,angle+2]]))
    for c in out:
        c['slope_group']='low' if c['angle']<=10 else 'medium' if c['angle']<=20 else 'high'
        c['slope_group_range_deg']={'low':[0,10],'medium':[10,20],'high':[20,40]}[c['slope_group']]
        c['slope_group_basis']='base piste / approach slope; bump faces, kicker upturn and runout are explicit local exceptions'
    return out

def profile_height(profile,x):
    """C2 height with monotonically interpolated slope inside each station interval."""
    x=np.asarray(x);a0,angle=profile[0];z0=8.;m0=-math.tan(math.radians(angle));result=z0+(x-a0)*m0
    for (a,aa),(b,ab) in zip(profile[:-1],profile[1:]):
        ma=-math.tan(math.radians(aa));mb=-math.tan(math.radians(ab));z1=z0+(b-a)*(ma+mb)/2
        result=np.where(x>=a,hermite(x,a,b,z0,z1,ma,mb),result)
        result=np.where(x>=b,z1+(x-b)*mb,result);z0=z1
    return result

def design_height(c,x,y):
    x,y=np.broadcast_arrays(x,y)
    s=-math.tan(math.radians(c['angle']))
    z=profile_height(c['slope_profile'],x) if 'slope_profile' in c else 8+s*x
    if c['kind']=='jump':
        za=float(profile_height(c['slope_profile'],16));s0=-math.tan(math.radians(c['slope_profile'][-1][1]));ml=math.tan(math.radians(c['lip_angle_deg']));sl=-math.tan(math.radians(c['landing_angle_deg']));zl=za+3*(s0+ml)
        z=np.where(x<16,z,hermite(x,16,22,za,zl,s0,ml))
        # An intentional sharp lip leading to a filled, descending tabletop deck.
        z=np.where(x>=22,zl-.04*(x-22),z)
        deck=c['table_end']; start=deck+2; zd=zl-.04*(deck-22)
        z=np.where(x>=deck,hermite(x,deck,start,zd,zd+sl, -.04,sl),z)
        zs=zd+sl
        z=np.where(x>=start,zs+sl*(x-start),z)
        z34=zs+sl*(34-start)
        z=np.where(x>=34,hermite(x,34,42,z34,z34+sl*4,sl,0),z)
        z=np.where(x>=42,z34+sl*4,z)
    else:
        if 'slope_profile' not in c:
            a=c['length']-18; b=c['length']-8; za=8+s*a
            z=np.where(x>=a,hermite(x,a,b,za,za+s*5,s,0),z)
            z=np.where(x>=b,za+s*5,z)
        if c['kind']=='cross':
            z=z+.045*y*smooth((x-8)/8)*(1-smooth((x-35)/8))
        if c['kind'] in ['single','row','moguls','random']:
            rng=np.random.default_rng(c['seed'])
            rows=[14] if c['kind']=='single' else np.arange(10,29,c['row_spacing'])
            for i,cx in enumerate(rows):
                ys=[0] if c['kind'] in ['single','row'] else np.arange(-4.8,5,c['column_spacing'])+(i%2)*c['stagger']
                for cy in ys:
                    a=c['amplitude']; rx=c['rx']; ry=c['ry']
                    if c['kind']=='random':
                        cxj=cx+rng.uniform(-.18,.18); cy+=rng.uniform(-.15,.15); a*=rng.uniform(.8,1.2)
                    else: cxj=cx
                    q=((x-cxj)/rx)**2+((y-cy)/ry)**2
                    z+=a*np.maximum(1-q,0)**3*smooth((x-6)/3)*(1-smooth((x-29)/3))
    return z

def base_model(name):
    p=CFG['physics']; root=ET.Element('mujoco',model=name)
    el(root,'compiler',angle='radian',autolimits='true')
    el(root,'option',timestep=p['timestep'],gravity='0 0 -9.81',integrator=p['integrator'],solver=p['solver'],cone=p['cone'],iterations=p['iterations'],tolerance=p['tolerance'])
    visual=el(root,'visual'); el(visual,'global',offwidth=1400,offheight=1000)
    el(visual,'headlight',ambient='.32 .34 .36',diffuse='.38 .38 .38',specular='.05 .05 .05')
    el(visual,'quality',shadowsize=2048)
    asset=el(root,'asset')
    el(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.30 .52 .73',rgb2='.86 .92 .97',width='128',height='768')
    for n,rgba in [('snow','.84 .91 .97 1'),('ski_blue','.02 .25 .46 1'),('ski_orange','1 .32 .055 1'),('edge','.25 .32 .37 1'),('binding','.08 .10 .13 1'),('red','.95 .09 .12 1'),('blue','.04 .3 .85 1')]:
        el(asset,'material',name=n,rgba=rgba,specular='.18',shininess='.2')
    world=el(root,'worldbody'); el(world,'light',directional='true',pos='0 0 30',dir='.2 1 -.45',diffuse='.95 .94 .92')
    return root,asset,world

def contact_attrs():
    p=CFG['physics']
    return dict(condim='3',friction='.05 .0 .0',solref=fmt(p['solref']),solimp=fmt(p['solimp']),margin='0',gap='0')

def course(c):
    root,asset,world=base_model(c['name'])
    if c['kind']=='plane':
        a=math.radians(c['angle']); el(world,'geom',name='snow_surface',type='plane',size='50 20 .1',quat=fmt([math.cos(a/2),0,math.sin(a/2),0]),material='snow',contype='1',conaffinity='6',**contact_attrs())
        c['z_origin']=0; c['triangles']=0
    else:
        x=np.linspace(0,c['length'],round(c['length']/c['dx'])+1); y=np.linspace(-c['width']/2,c['width']/2,round(c['width']/c['dy'])+1)
        z=design_height(c,x[None,:],y[:,None]); low=float(z.min()); scale=max(float(z.max()-low),.01)
        h=((z-low)/scale).astype('<f4')
        file=ROOT/'terrain'/f"{c['name']}.bin"
        file.write_bytes(struct.pack('<ii',len(y),len(x))+h.tobytes())
        np.savez_compressed(ROOT/'terrain'/f"{c['name']}.npz",x=x,y=y,z=low+scale*h.astype(float))
        el(asset,'hfield',name='terrain',file=f"../terrain/{c['name']}.bin",size=fmt([c['length']/2,c['width']/2,scale,.5]))
        el(world,'geom',name='snow_surface',type='hfield',hfield='terrain',pos=fmt([c['length']/2,0,low]),material='snow',contype='1',conaffinity='6',**contact_attrs())
        c.update(z_origin=low,height_scale=scale,rows=len(y),columns=len(x),triangles=2*(len(y)-1)*(len(x)-1))
    gates=[]
    if c['kind']=='piste':
        for i,x in enumerate(range(16,73,8)):
            cy=3*(-1)**i; width=5
            gates.append(dict(id=i,center=[x,cy,float(design_height(c,x,cy))],width=width,pass_direction=[1,0],normal=[1,0,0]))
            for j,y in enumerate([cy-width/2,cy+width/2]):
                z=float(design_height(c,x,y)); mat='red' if i%2==0 else 'blue'
                el(world,'geom',name=f'gate_{i}_{j}',type='cylinder',pos=fmt([x,y,z+.7]),size='.018 .7',material=mat,contype='0',conaffinity='0',group='2')
                el(world,'geom',type='box',pos=fmt([x,y+(.16 if j==0 else -.16),z+1.12]),size='.008 .16 .16',material=mat,contype='0',conaffinity='0',group='2')
    if c['kind']!='plane':
        for x in np.arange(2,c['length'],8):
            for y in [-c['width']/2+.3,c['width']/2-.3]:
                z=float(design_height(c,x,y)); el(world,'geom',type='capsule',fromto=fmt([x,y,z,x,y,z+.65]),size='.018',material='blue',contype='0',conaffinity='0',group='2')
    c['gates']=gates; c['material_id']='packed_snow_v1'; c['spawn_xy']=[2,0]
    c['finish_x']=c['length']-3; c['boundary_policy']='terminate on finite XY domain exit; no walls; planes have logical bounds only'
    c['region_intervals']=([[0,16,'approach'],[16,22,'takeoff'],[22,c['table_end'],'flight_gap'],[c['table_end'],34,'landing'],[34,64,'runout']] if c['kind']=='jump' else [[0,6,'approach'],[6,32,'moguls'],[32,c['length'],'runout']] if c['kind'] in ['single','row','moguls','random'] else [[0,c['length']-18,'approach'],[c['length']-18,c['length'],'runout']])
    if c['kind']=='jump': c['flight_gap_semantics']='filled tabletop; collision remains present; region is spatial, never an airborne flag'
    (ROOT/'terrain'/f"{c['name']}.json").write_text(json.dumps(c,indent=2)+'\n')
    save_xml(root,ROOT/'scenes'/f"{c['name']}.xml")
    return c

def shape(x):
    p=CFG['ski']; r=np.abs(np.asarray(x))/(p['length']/2)
    end=np.where(np.asarray(x)>=0,p['width_tip'],p['width_tail'])
    w=p['width_waist']+(end-p['width_waist'])*r*r
    rocker=np.where(np.asarray(x)>=0,p['tip_rise'],p['tail_rise'])*smooth((r-.7)/.3)
    z=p['camber']*(1-r*r)+rocker
    return w,z

def prism_mesh(xs, center, inset=0):
    p=CFG['ski']; verts=[]
    for x in xs:
        w,z=shape(x)
        for y,dz in [(-w/2+inset,-p['thickness']/2),(w/2-inset,-p['thickness']/2),(w/2-inset,p['thickness']/2),(-w/2+inset,p['thickness']/2)]:
            verts.append([x-center,y,z+dz])
    faces=[(0,2,1),(0,3,2)]; k=4*(len(xs)-1); faces +=[(k,k+1,k+2),(k,k+2,k+3)]
    for i in range(len(xs)-1):
        for j in range(4):
            a=4*i+j;b=4*i+(j+1)%4;d=a+4;c=b+4
            faces +=[(a,b,c),(a,c,d)]
    return verts,faces

def obj(path,verts,faces):
    path.write_text(''.join('v '+fmt(v)+'\n' for v in verts)+''.join('f '+' '.join(str(i+1) for i in f)+'\n' for f in faces))

def ski_assets(asset):
    p=CFG['ski']; n=p['segments']; d=p['length']/n
    for i in range(n):
        cx=(i-(n-1)/2)*d
        v,f=prism_mesh(np.linspace(cx-d/2,cx+d/2,9),cx)
        obj(ROOT/'meshes'/f'ski_visual_{i}.obj',v,f)
        el(asset,'mesh',name=f'ski_visual_{i}',file=f'../meshes/ski_visual_{i}.obj')
        for j in range(2):
            xa=cx-d/2+j*d/2; xb=xa+d/2
            # Bottom strip leaves 4 mm clear for each discrete steel edge.
            v,f=prism_mesh([xa,xb],cx,inset=.004)
            obj(ROOT/'meshes'/f'ski_base_{i}_{j}.obj',v,f)
            el(asset,'mesh',name=f'ski_base_{i}_{j}',file=f'../meshes/ski_base_{i}_{j}.obj')

def ski_body(parent,prefix,pos=(0,0,0),free=False):
    p=CFG['ski']; n=p['segments']; d=p['length']/n; mid=n//2
    root=el(parent,'body',name=f'{prefix}_mount',pos=fmt(pos))
    if free: el(root,'freejoint',name=f'{prefix}_free')
    mass=p['binding_mass']; el(root,'inertial',pos='0 0 .025',mass=mass,diaginertia=fmt([mass*(.065**2+.04**2)/12,mass*(.24**2+.04**2)/12,mass*(.24**2+.065**2)/12]))
    el(root,'geom',name=f'{prefix}_binding',type='box',pos='0 0 .025',size='.12 .0325 .02',material='binding',contype='0',conaffinity='0',mass='0',group='2')
    el(root,'site',name=f'{prefix}_mount_frame',pos='0 0 .045',size='.01',group='5')
    bodies={}; meta=[]
    for i in [mid,*range(mid+1,n),*range(mid-1,-1,-1)]:
        cx=(i-mid)*d
        if i==mid: par=root; origin=0; rel=0
        else:
            parent_i=i-1 if i>mid else i+1; par=bodies[parent_i][0]
            origin=cx+(-d/2 if i>mid else d/2); rel=origin-bodies[parent_i][1]
        b=el(par,'body',name=f'{prefix}_segment_{i}',pos=fmt([rel,0,0])); bodies[i]=(b,origin)
        if i!=mid: el(b,'joint',name=f'{prefix}_flex_{i}',type='hinge',axis='0 1 0',stiffness=p['EI']/d,damping=p['hinge_damping'],range=fmt([-p['hinge_limit'],p['hinge_limit']]),armature='0',frictionloss='0')
        w,z=shape(cx); mass=p['blade_mass']/n
        el(b,'inertial',pos=fmt([cx-origin,0,z]),mass=mass,diaginertia=fmt([mass*(w*w+p['thickness']**2)/12,mass*(d*d+p['thickness']**2)/12,mass*(d*d+w*w)/12]))
        el(b,'geom',name=f'{prefix}_visual_{i}',type='mesh',mesh=f'ski_visual_{i}',pos=fmt([cx-origin,0,0]),material='ski_blue' if i%2==0 else 'ski_orange',contype='0',conaffinity='0',mass='0',group='2')
        for j in range(2):
            xa=cx-d/2+j*d/2; xb=xa+d/2; wa,za=shape(xa); wb,zb=shape(xb)
            name=f'{prefix}_base_{i}_{j}'
            el(b,'geom',name=name,type='mesh',mesh=f'ski_base_{i}_{j}',pos=fmt([cx-origin,0,0]),rgba='.1 .1 .12 0',contype='2',conaffinity='7',mass='0',group='3',**contact_attrs())
            meta.append(dict(geom=name,body=f'{prefix}_segment_{i}',kind='base',axis=[1,0,0],side=0))
            for side in [-1,1]:
                # Capsule is inside the rendered edge to within its 2 mm radius.
                a=np.array([xa-origin,side*(wa/2-.002),za-.004]); end=np.array([xb-origin,side*(wb/2-.002),zb-.004])
                name=f'{prefix}_edge_{i}_{j}_{"L" if side==1 else "R"}'
                el(b,'geom',name=name,type='capsule',fromto=fmt([*a,*end]),size=p['contact_radius'],material='edge',contype='2',conaffinity='7',mass='0',group='3',**contact_attrs())
                tangent=(end-a)/np.linalg.norm(end-a)
                meta.append(dict(geom=name,body=f'{prefix}_segment_{i}',kind='edge',axis=tangent.tolist(),side=side))
    return root,meta

def main():
    for folder in ['scenes','meshes','terrain','reports']: (ROOT/folder).mkdir(exist_ok=True)
    v,f=prism_mesh(np.linspace(-CFG['ski']['length']/2,CFG['ski']['length']/2,57),0)
    obj(ROOT/'meshes/ski_complete_rest.obj',v,f)
    courses=[course(c) for c in definitions()]
    for name,positions in [('ski_single',[(0,0,.12)]),('ski_pair',[(0,.13,.12),(0,-.13,.12)])]:
        root,asset,world=base_model(name); ski_assets(asset)
        el(world,'geom',name='snow_surface',type='plane',size='8 4 .1',material='snow',contype='1',conaffinity='6',**contact_attrs())
        meta=[]
        for i,pos in enumerate(positions): meta+=ski_body(world,f'ski{i}',pos,True)[1]
        save_xml(root,ROOT/'scenes'/f'{name}.xml')
        (ROOT/'configs'/f'{name}_contacts.json').write_text(json.dumps(meta,indent=2)+'\n')
    manifest={'version':CFG['version'],'courses':courses,'ski':CFG['ski'],'ski_visual_triangles_per_blade':7*68,'contact_geoms_per_blade':42,'passive_joints_per_blade':6,'physics':CFG['physics']}
    curriculum={'version':CFG['version'],'group_boundary_ownership':'10 degrees belongs to low, 20 degrees belongs to medium; no duplicate sampling at boundaries','groups':{group:[c['name'] for c in courses if c['slope_group']==group] for group in ['low','medium','high']},'special_terrain_rule':'single bump, bump row, moguls and tabletop scenes occur only in high group','control_rates_hz':{'physics':1000,'LLC':100,'HLC':10},'phase_rule':'region(x,y) is spatial only. Airborne/landing/recovery phases require contact/state estimation.','promotion_rule':'Do not sample high terrain for policy training before physical validation and skill readiness.','training_ready':False}
    (ROOT/'configs/curriculum.json').write_text(json.dumps(curriculum,indent=2)+'\n')
    manifest['files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scenes','meshes','terrain','configs'] for p in sorted((ROOT/folder).glob('*')) if p.is_file()}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {len(courses)} courses, two ski fixtures; max terrain triangles {max(c["triangles"] for c in courses)}')

if __name__=='__main__': main()
