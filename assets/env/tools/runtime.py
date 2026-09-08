"""CPU reference adapters. XML alone has isotropic fallback friction, not skiing physics."""
import json
import math
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]

class Terrain:
    """Vectorized interpolation of MuJoCo's actual heightfield triangles, not bilinear."""
    def __init__(self,name):
        self.meta=json.loads((ROOT/'terrain'/f'{name}.json').read_text())
        if self.meta['kind']!='plane':
            data=np.load(ROOT/'terrain'/f'{name}.npz'); self.x=data['x']; self.y=data['y']; self.z=data['z']

    def sample(self,x,y):
        x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float)); c=self.meta
        valid=(x>=0)&(x<=c['length'])&(np.abs(y)<=c['width']/2)&np.isfinite(x)&np.isfinite(y)
        if c['kind']=='plane':
            slope=-math.tan(math.radians(c['angle'])); z=slope*x; gx=np.full(x.shape,slope); gy=np.zeros(x.shape)
        else:
            xx=np.nan_to_num(x); yy=np.nan_to_num(y)
            u=np.clip(xx,0,c['length'])/c['dx']; v=(np.clip(yy,-c['width']/2,c['width']/2)+c['width']/2)/c['dy']
            i=np.minimum(u.astype(int),len(self.x)-2); j=np.minimum(v.astype(int),len(self.y)-2); a=u-i; b=v-j
            z00=self.z[j,i];z10=self.z[j,i+1];z01=self.z[j+1,i];z11=self.z[j+1,i+1]
            # MuJoCo splits each cell from (0,0) to (1,1).
            lower=a>=b
            gx=np.where(lower,z10-z00,z11-z01)/c['dx']
            gy=np.where(lower,z11-z10,z01-z00)/c['dy']
            z=np.where(lower,z00+a*(z10-z00)+b*(z11-z10),z00+a*(z11-z01)+b*(z01-z00))
        n=np.stack([-gx,-gy,np.ones(x.shape)],axis=-1); n/=np.linalg.norm(n,axis=-1,keepdims=True)
        return np.where(valid,z,np.nan),np.where(valid[...,None],n,np.nan),valid

    def height(self,x,y): return self.sample(x,y)[0]
    def normal(self,x,y): return self.sample(x,y)[1]
    def region(self,x,y):
        x,y=np.broadcast_arrays(np.asarray(x),np.asarray(y)); result=np.full(x.shape,'out_of_bounds',dtype='<U16')
        valid=self.sample(x,y)[2]
        for a,b,name in self.meta['region_intervals']: result=np.where(valid&(x>=a)&(x<=b),name,result)
        return result
    def material(self,x,y): return np.where(self.sample(x,y)[2],self.meta['material_id'],'invalid')
    def boundary_distances(self,x,y): return np.array([y+self.meta['width']/2,self.meta['width']/2-y,x,self.meta['length']-x])
    def gate_crossing(self,previous,current,index):
        """Ordered task code advances index only after True; segment-plane interpolation."""
        gates=self.meta['gates']
        if not 0<=index<len(gates): return False
        gate=gates[index]; gx,gy,_=gate['center']; p=np.asarray(previous); q=np.asarray(current)
        if not p[0]<gx<=q[0]: return False
        t=(gx-p[0])/(q[0]-p[0]); yy=p[1]+t*(q[1]-p[1])
        return bool(abs(yy-gy)<=gate['width']/2)

class SnowStepper:
    """3.3.7 CPU split step with constraint rebuild after local friction-frame edits.

    No applied body forces. Caller sets ctrl before step. Contact diagnostics belong
    to the pre-integration state; do not call mj_forward before reading them.
    """
    def __init__(self,model,metadata,physics=None):
        if mujoco.__version__!='3.3.7': raise RuntimeError('Revalidate pipeline before changing MuJoCo version')
        self.m=model
        if model.opt.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST: raise ValueError('Requires implicitfast')
        self.p=physics or json.loads((ROOT/'configs/design.json').read_text())['physics']
        self.items={model.geom(item['geom']).id:item for item in metadata}
        self.snow=model.geom('snow_surface').id
        self.degenerate_count=0

    def prepare(self,d):
        m=self.m; mujoco.mj_step1(m,d)
        for k in range(d.ncon):
            c=d.contact[k]; g1,g2=map(int,c.geom)
            if self.snow not in [g1,g2]: continue
            gid=g2 if g1==self.snow else g1
            item=self.items.get(gid)
            if item is None: continue
            R=d.xmat[m.geom_bodyid[gid]].reshape(3,3)
            n=c.frame[:3].copy(); axis=R@np.asarray(item['axis']); t=axis-n*np.dot(axis,n); length=np.linalg.norm(t)
            if length<1e-8:
                self.degenerate_count+=1
                basis=np.eye(3)[np.argmin(np.abs(n))]; t=basis-n*np.dot(basis,n); length=np.linalg.norm(t)
            t/=length; b=np.cross(n,t)
            c.frame[3:6]=t; c.frame[6:9]=b
            # A roll angle around local board X. Pitch alone cannot activate an edge.
            up=n if g1==self.snow else -n
            angle=math.atan2(float(np.dot(up,R[:,1])),float(np.dot(up,R[:,2])))
            u=np.clip((abs(angle)-math.radians(self.p['edge_on_deg']))/math.radians(self.p['edge_full_deg']-self.p['edge_on_deg']),0,1)
            # Lower edge has side * dot(n, local_y) < 0. Inverted skis do not carve.
            bearing=item['kind']=='edge' and item['side']*np.dot(up,R[:,1])<0 and np.dot(up,R[:,2])>0
            activation=u*u*(3-2*u) if bearing else 0
            c.friction[:]=[self.p['mu_parallel'],self.p['mu_flat']+activation*(self.p['mu_edge']-self.p['mu_flat']),0,0,0]
        mujoco.mj_makeConstraint(m,d); mujoco.mj_island(m,d); mujoco.mj_projectConstraint(m,d)
        mujoco.mj_fwdVelocity(m,d)

    def step(self,d):
        self.prepare(d)
        mujoco.mj_step2(self.m,d)

    def contacts(self,d):
        result=[]
        for i in range(d.ncon):
            c=d.contact[i]
            if self.snow not in c.geom: continue
            f=np.zeros(6); mujoco.mj_contactForce(self.m,d,i,f)
            result.append(dict(geom=[int(g) for g in c.geom],position=c.pos.copy().tolist(),frame=c.frame.copy().tolist(),force_contact_frame=f.tolist(),distance=float(c.dist)))
        return result
