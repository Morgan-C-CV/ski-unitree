"""Exposed-surface sphere-sample collision; all forces remain native MuJoCo constraints.

Convex ski volumes still handle ski/robot collisions. Snow support is represented by
explicit small spheres on the base/edges. Nearest points are computed against actual
heightfield triangles, never against the hidden sides of their internal prisms.
"""
import math
import numpy as np
import mujoco


def closest_triangles(points, triangles):
    """Closest point on each triangle; points [P,3], triangles [P,T,3,3]."""
    a,b,c=triangles[:,:,0],triangles[:,:,1],triangles[:,:,2]
    ab=b-a;ac=c-a;normal=np.cross(ab,ac);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    p=points[:,None,:];signed=np.sum((p-a)*normal,axis=-1)
    projection=p-signed[...,None]*normal;v=projection-a
    aa=np.sum(ab*ab,axis=-1);bb=np.sum(ab*ac,axis=-1);cc=np.sum(ac*ac,axis=-1)
    av=np.sum(ab*v,axis=-1);cv=np.sum(ac*v,axis=-1);den=aa*cc-bb*bb
    u=(cc*av-bb*cv)/den;w=(aa*cv-bb*av)/den
    interior=(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)
    candidates=[projection]
    for start,end in [(a,b),(b,c),(c,a)]:
        direction=end-start;t=np.clip(np.sum((p-start)*direction,axis=-1)/np.sum(direction*direction,axis=-1),0,1)
        candidates.append(start+t[...,None]*direction)
    candidates=np.stack(candidates,axis=2)
    distances=np.sum((candidates-p[:,:,None,:])**2,axis=-1)
    distances[:,:,0]=np.where(interior,distances[:,:,0],np.inf)
    choice=np.argmin(distances,axis=2);row=np.arange(len(points))[:,None];col=np.arange(triangles.shape[1])[None,:]
    q=candidates[row,col,choice];ds=np.min(distances,axis=2)
    nearest=np.argmin(ds,axis=1)
    return q[np.arange(len(points)),nearest],normal[np.arange(len(points)),nearest]


class SurfaceContacts:
    def __init__(self,m,metadata,design,stations=32):
        self.m=m;self.snow=m.geom('snow_surface').id
        self.items={m.geom(it['geom']).id:it for it in metadata}
        # Snow narrow phase is supplied here; preserve all robot/ski volume pairs.
        if m.geom_contype[self.snow]!=1 or any(m.geom_contype[g]!=2 for g in self.items):
            raise ValueError('Expected snow=1, ski=2, robot=4 collision bit contract')
        m.geom_conaffinity[self.snow] &= ~2
        for gid in self.items:m.geom_conaffinity[gid] &= ~1
        self.samples=[];self.tracks=[]
        ski=design['ski'];n=ski['segments'];length=ski['length'];segment=length/n;mid=n//2
        # One representative geom per body/role. Samples are collision quadrature
        # points, not additional rigid bodies, masses or active degrees of freedom.
        bodies={}
        for gid,it in self.items.items():
            key=(m.geom_bodyid[gid],it['kind'],it['side'])
            bodies.setdefault(key,gid)
        for (bid,kind,side),gid in bodies.items():
            index=int(m.body(bid).name.rsplit('_',1)[1]);cx=(index-mid)*segment
            origin=0 if index==mid else cx+(-segment/2 if index>mid else segment/2)
            for j in range(stations):
                x=cx-segment/2+j*segment/(stations-1)
                r=abs(x)/(length/2);tip=x>=0
                end=ski['width_tip'] if tip else ski['width_tail'];width=ski['width_waist']+(end-ski['width_waist'])*r*r
                dr=(1 if tip else -1)/(length/2);dw=2*(end-ski['width_waist'])*r*dr
                u=np.clip((r-.7)/.3,0,1);ss=u**3*(10-15*u+6*u*u)
                rise=ski['tip_rise'] if tip else ski['tail_rise'];z=ski['camber']*(1-r*r)+rise*ss
                dz=-2*ski['camber']*r*dr+rise*30*u*u*(1-u)**2*dr/.3
                if kind=='edge':
                    radius=ski['contact_radius'];ys=[side*(width/2-radius)];axis=np.array([1,side*dw/2,dz])
                else:
                    radius=.001;ys=[-width*.25,0,width*.25];axis=np.array([1,0,0])
                for yi,y in enumerate(ys):
                    local=[x-origin,y,z-ski['thickness']/2+radius]
                    self.samples.append((bid,gid,local,radius,axis/np.linalg.norm(axis)))
                    self.tracks.append((bid,kind,side,yi))
                    # Upper skin supports inverted falls; inverted poses never activate carving grip.
                    top=[x-origin,y,z+ski['thickness']/2-radius]
                    self.samples.append((bid,gid,top,radius,axis/np.linalg.norm(axis)))
                    self.tracks.append((bid,kind,side,yi+len(ys)))
        self.bids=np.array([x[0] for x in self.samples]);self.gids=np.array([x[1] for x in self.samples]);self.local=np.array([x[2] for x in self.samples]);self.radii=np.array([x[3] for x in self.samples]);self.axes=np.array([x[4] for x in self.samples])
        if m.geom_type[self.snow]==mujoco.mjtGeom.mjGEOM_HFIELD:
            hid=m.geom_dataid[self.snow];nr=int(m.hfield_nrow[hid]);nc=int(m.hfield_ncol[hid]);self.size=m.hfield_size[hid].copy()
            self.z=m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid]+nr*nc].reshape(nr,nc).astype(float)*self.size[2]
            self.dx=2*self.size[0]/(nc-1);self.dy=2*self.size[1]/(nr-1)
            gx=np.max(np.abs(np.diff(self.z,axis=1)))/self.dx
            gy=np.max(np.abs(np.diff(self.z,axis=0)))/self.dy
            self.slope_bound=float(np.hypot(gx,gy))
        self.track_indices=np.array([[i for i,t in enumerate(self.tracks) if t==key] for key in dict.fromkeys(self.tracks)])
        self.added_axes={};self.max_contacts=0

    def geometry(self,d,points):
        g=self.snow;R=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g]
        local=np.einsum('ni,ij->nj',points-pos,R)
        if self.m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:
            normals=np.broadcast_to(R[:,2],points.shape).copy();signed=local[:,2]
            closest=points-signed[:,None]*normals;valid=np.ones(len(points),bool)
            return closest,normals,signed,valid
        u=(local[:,0]+self.size[0])/self.dx;v=(local[:,1]+self.size[1])/self.dy
        nc=self.z.shape[1];nr=self.z.shape[0]
        ii=np.clip(np.floor(u).astype(int),0,nc-2);jj=np.clip(np.floor(v).astype(int),0,nr-2)
        aa=u-ii;bb=v-jj
        z00=self.z[jj,ii];z10=self.z[jj,ii+1];z01=self.z[jj+1,ii];z11=self.z[jj+1,ii+1]
        height=np.where(aa>=bb,z00+aa*(z10-z00)+bb*(z11-z10),z00+aa*(z11-z01)+bb*(z01-z00))
        inside_xy=(u>=0)&(u<=nc-1)&(v>=0)&(v<=nr-1)
        inside=inside_xy&(local[:,2]<height)
        # Lipschitz bound conservatively rejects spheres above the entire local
        # surface neighborhood. Far values are positive bounds, not contact normals.
        near=(local[:,2]-height <= self.radii*np.sqrt(1+self.slope_bound**2)+1e-8)
        indices=np.flatnonzero(near)
        nearest_all=local.copy();nearest_all[:,2]=height
        normals_all=np.zeros_like(local);normals_all[:,2]=1
        signed_all=(local[:,2]-height)/np.sqrt(1+self.slope_bound**2)
        valid=(np.abs(local[:,0])<=self.size[0]+self.radii)&(np.abs(local[:,1])<=self.size[1]+self.radii)
        if not len(indices):
            return np.einsum('ni,ji->nj',nearest_all,R)+pos,np.einsum('ni,ji->nj',normals_all,R),signed_all,valid
        offsets=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
        i=np.clip(ii[indices,None]+offsets[:,0],0,nc-2);j=np.clip(jj[indices,None]+offsets[:,1],0,nr-2)
        x=-self.size[0]+i*self.dx;y=-self.size[1]+j*self.dy
        a=np.stack([x,y,self.z[j,i]],-1);b=np.stack([x+self.dx,y,self.z[j,i+1]],-1)
        c=np.stack([x+self.dx,y+self.dy,self.z[j+1,i+1]],-1);e=np.stack([x,y+self.dy,self.z[j+1,i]],-1)
        triangles=np.concatenate([np.stack([a,b,c],axis=2),np.stack([a,c,e],axis=2)],axis=1)
        nearest,face_normals=closest_triangles(local[indices],triangles)
        delta=local[indices]-nearest;dist=np.linalg.norm(delta,axis=-1);sign=np.where(inside[indices],-1.,1.)
        normal=np.where((dist>1e-10)[:,None],sign[:,None]*delta/np.maximum(dist,1e-10)[:,None],face_normals)
        valid=(np.abs(local[:,0])<=self.size[0]+self.radii)&(np.abs(local[:,1])<=self.size[1]+self.radii)
        nearest_all[indices]=nearest;normals_all[indices]=normal;signed_all[indices]=sign*dist
        return np.einsum('ni,ji->nj',nearest_all,R)+pos,np.einsum('ni,ji->nj',normals_all,R),signed_all,valid

    def build(self,d):
        # Exclude rather than delete native records, preserving unrelated contacts.
        # Excluded records have no constraint/force; diagnostics must skip them.
        for contact in d.contact[:d.ncon]:
            a,b=map(int,contact.geom)
            if self.snow in [a,b] and (b if a==self.snow else a) in self.items:contact.exclude=1
        rotations=d.xmat[self.bids].reshape(-1,3,3)
        points=np.einsum('nij,nj->ni',rotations,self.local)+d.xpos[self.bids]
        q,n,signed,valid=self.geometry(d,points)
        distances=signed-self.radii;self.added_axes={}
        enabled=(self.m.geom_contype[self.gids]!=0)|(self.m.geom_conaffinity[self.gids]!=0)
        # A manifold contains local distance minima, not every penetrating sample.
        # Otherwise increasing tessellation adds redundant compliant constraints.
        selected=np.zeros(len(points),bool)
        dd=distances[self.track_indices]
        pad=np.full((len(dd),1),np.inf)
        minima=(dd<=np.concatenate([pad,dd[:,:-1]],axis=1))&(dd<=np.concatenate([dd[:,1:],pad],axis=1))
        selected[self.track_indices[minima]]=True
        for index in np.flatnonzero(valid&enabled&selected&(distances<=0)):
            gid=int(self.gids[index]);c=mujoco.MjContact();c.geom[:]=[self.snow,gid];c.dim=3;c.dist=distances[index]
            c.pos[:]=points[index]-(self.radii[index]+.5*distances[index])*n[index]
            c.frame[:3]=n[index]
            # SnowStepper sets the native friction frame and coefficients afterwards.
            c.solref[:]=self.m.geom_solref[self.snow];c.solimp[:]=self.m.geom_solimp[self.snow]
            c.friction[:]=[.05,.08,0,0,0];c.includemargin=0;c.exclude=0
            address=d.ncon
            if mujoco.mj_addContact(self.m,d,c)!=0:raise RuntimeError('Contact arena exhausted')
            self.added_axes[address]=self.axes[index]
        self.max_contacts=max(self.max_contacts,len(self.added_axes))
