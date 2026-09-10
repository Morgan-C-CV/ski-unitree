"""Ground-to-ground G1 jump task. No per-step state overrides or applied root forces."""
from pathlib import Path
import json, math, sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'assets/env/tools'))
from runtime import ROOT,Terrain,SnowStepper
from annotate_720 import ids
from cross_720 import grab_points

class SkiJumpEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':50}
    def __init__(self,stage='landing',target_turns=0.,frame_index=1,initial_speed=7.05253,render_mode=None):
        super().__init__()
        if stage not in ('landing','full'):raise ValueError(stage)
        self.stage=stage;self.target_turns=float(target_turns);self.frame_index=frame_index;self.initial_speed=initial_speed;self.render_mode=render_mode
        self.m=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/g1_tabletop_steep.xml'));self.d=mujoco.MjData(self.m)
        self.snow=SnowStepper(self.m,json.loads((ROOT/'configs/g1_contacts.json').read_text()));self.terrain=Terrain('tabletop_steep')
        self.qa,self.va,self.names=ids(self.m);self.jids=self.m.actuator_trnid[:,0];self.lo=self.m.jnt_range[self.jids,0];self.hi=self.m.jnt_range[self.jids,1]
        self.passive=np.array([j for j in range(self.m.njnt) if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and j not in self.jids]);self.fq=self.m.jnt_qposadr[self.passive];self.fv=self.m.jnt_dofadr[self.passive]
        self.frames=json.loads((PROJECT/'docs/imgs/keyframe/keyframes.json').read_text())['frames'];self.start=np.array(self.frames[frame_index]['qpos'])
        ik=json.loads((PROJECT/'docs/imgs/keyframe/cross_grab_pose_ik.json').read_text());self.grab=np.array([ik['joints'][n] for n in self.names])
        self.skis=[self.m.body(side+'_ski_segment_3').id for side in ('left','right')]
        self.dt=.01;self.substeps=round(self.dt/self.m.opt.timestep)
        assert abs(self.substeps*self.m.opt.timestep-self.dt)<1e-12
        self.mass=float(self.m.body_subtreemass[1]);self.weight=self.mass*9.81
        self.action_space=spaces.Box(-1,1,(29,),np.float32);self.action_scale=np.array([.55 if ('hip' in n or 'waist' in n) else .4 for n in self.names])
        self.renderer=None
        self.reset(seed=0)
        self.observation_space=spaces.Box(-np.inf,np.inf,self._obs().shape,np.float32)

    def _forward(self):
        mujoco.mj_forward(self.m,self.d);mujoco.mj_subtreeVel(self.m,self.d)

    def _clearance(self):
        s=self.snow.surface;p=self.d.xpos[s.bids]+np.einsum('nij,nj->ni',self.d.xmat[s.bids].reshape(-1,3,3),s.local)
        return float(np.min(p[:,2]-s.radii-self.terrain.height(p[:,0],p[:,1])))

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};self.randomize=bool(options.get('randomize',True))
        mujoco.mj_resetData(self.m,self.d);self.d.qpos[:]=self.start
        self.stage=options.get('stage',self.stage);self.target_turns=float(options.get('target_turns',self.target_turns))
        if self.stage=='landing':
            self.d.qpos[:3]=[float(self.np_random.uniform(27,29)) if self.randomize else 28,0,0];self.d.qpos[3:7]=[1,0,0,0];self.d.qpos[7:]=0
            self.d.qpos[self.qa]=self._neutral(self.d.qpos[0],1.4)
            self._forward();self.d.qpos[2]+=(float(self.np_random.uniform(.015,.055)) if self.randomize else .035)-self._clearance()
        self._forward()
        speed=float(options.get('initial_speed',self.initial_speed))
        if self.randomize:speed+=float(self.np_random.uniform(-.25,.25))
        normal=self.terrain.normal(self.d.qpos[0],0);tangent=np.array([normal[2],0,-normal[0]])
        self.d.qvel[:]=0;self.d.qvel[:3]=speed*tangent
        if self.stage=='landing':self.d.qvel[:3]-=normal*(float(self.np_random.uniform(.1,1.0)) if self.randomize else .5)
        self.d.ctrl[:]=self.d.qpos[self.qa];self._forward()
        self.elapsed=0.;self.airtime=0.;self.contact_time=0.;self.no_contact_time=0.;self.stable_time=0.;self.spin=0.;self.spin_at_touchdown=0.;self.cross_time=0.;self.palm_error_integral=0.
        self.phase='approach' if self.stage=='full' else 'airborne';self.launched=self.stage=='landing';self.seen_support=False;self.touched=False;self.success=False;self.fallen=False;self.failure_reason='';self.takeoff_x=None;self.takeoff_airtime=0.
        self.loads=np.zeros(2);self.previous_action=np.zeros(29);self.previous_target=self.d.qpos[self.qa].copy();self.last_yaw=self._yaw();self.episode_return=0.;self.max_penetration=0.;self.max_joint_torque_ratio=0.;self.start_x=float(self.d.qpos[0]);self.landing_x=None;self.total_airtime=0.;self.trick_bonus=0.
        return self._obs(),self._info()

    def _yaw(self):
        R=self.d.xmat[1].reshape(3,3);return math.atan2(R[1,0],R[0,0])

    def _neutral(self,x,knee=1.3):
        normal=self.terrain.normal(np.clip(x,0,63),0);pitch=math.atan2(normal[0],normal[2]);q=np.zeros(29)
        for side,sign in [('left',1),('right',-1)]:
            ankle=float(np.clip(pitch-.45*knee,-.8,.45));values={'knee_joint':knee,'hip_pitch_joint':pitch-knee-ankle,'ankle_pitch_joint':ankle,'shoulder_pitch_joint':-.25,'shoulder_roll_joint':sign*.7,'elbow_joint':.4}
            for key,value in values.items():q[self.names.index(side+'_'+key)]=value
        return q

    def _nominal(self):
        x=float(self.d.qpos[0]);q=self._neutral(x,1.4 if self.phase=='landing' else 1.3)
        if self.phase=='airborne' and self.stage=='full' and self.target_turns>0:
            # Reference shapes guide joints only; the root is always unconstrained.
            # Expand at low predicted ground clearance, based on current state only.
            h=float(self.d.subtree_com[1,2]-self.terrain.height(x,self.d.qpos[1]));vz=float(self.d.subtree_linvel[1,2]);blend=np.clip((h-.95)/.65,0,1) if vz<0 else 1.
            q=(1-blend)*q+blend*self.grab
        return q

    def _obs(self):
        R=self.d.xmat[1].reshape(3,3);x,y,z=self.d.qpos[:3];normal=self.terrain.normal(np.clip(x,0,64),np.clip(y,-8,8));j=(self.d.qpos[self.qa]-(self.lo+self.hi)/2)/((self.hi-self.lo)/2)
        fields=[j,self.d.qvel[self.va]/20,self.d.qpos[self.fq]/.3,self.d.qvel[self.fv]/20,R.T@np.array([0.,0.,-1.]),R.T@self.d.subtree_linvel[1]/10,self.d.qvel[3:6]/10,self.loads/self.weight,normal,
                np.array([(z-float(self.terrain.height(np.clip(x,0,64),np.clip(y,-8,8))))/2,(22-x)/5,y/8,self.elapsed/5,self.airtime/2,self.stable_time,self.spin/(4*np.pi),self.target_turns/2]),
                np.array([self.phase==p for p in ('approach','airborne','landing')],float),self.previous_action,(self.previous_target-(self.lo+self.hi)/2)/((self.hi-self.lo)/2),np.array([self.seen_support,self.touched,self.no_contact_time,self.contact_time],float)]
        for b in self.skis:
            fields.extend([(self.d.xpos[b]-self.d.subtree_com[1])@R,self.d.xmat[b].reshape(3,3)[:,0]@R,self.d.xmat[b].reshape(3,3)[:,2]@R])
        # Fixed world/downhill-aligned two-dimensional lookahead; all values are current terrain queries.
        dx=np.repeat([0.,1.,2.,4.,7.],3);dy=np.tile([-1.,0.,1.],5);height,_,valid=self.terrain.sample(x+dx,y+dy)
        fields.extend([np.nan_to_num((height-z)/5),valid.astype(float)])
        return np.concatenate(fields).astype(np.float32)

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('Expected 29 finite action values')
        a=np.clip(action,-1,1);target=np.clip(self._nominal()+self.action_scale*a,self.lo+.01,self.hi-.01)
        # Rate bound limits PD target jumps; policy observes last submitted action.
        target=np.clip(target,self.previous_target-.12,self.previous_target+.12);self.d.ctrl[:]=target
        load_impulse=np.zeros(2);body_collision=False;fall=False;executed=0
        for _ in range(self.substeps):
            assert not np.any(self.d.xfrc_applied) and not np.any(self.d.qfrc_applied)
            self.snow.step(self.d);executed+=1
            for c in self.snow.contacts(self.d):
                f=max(c['force_contact_frame'][0],0);gid=next(g for g in c['geom'] if g!=self.snow.snow);name=self.m.geom(gid).name
                if name.startswith('left_ski'):load_impulse[0]+=f*self.m.opt.timestep
                elif name.startswith('right_ski'):load_impulse[1]+=f*self.m.opt.timestep
                elif f>5:body_collision=True
                self.max_penetration=max(self.max_penetration,-c['distance'])
            limits=self.m.jnt_actfrcrange[self.jids,1];self.max_joint_torque_ratio=max(self.max_joint_torque_ratio,float(np.max(abs(self.d.qfrc_actuator[self.va])/limits)))
            if body_collision or not np.isfinite(self.d.qpos).all() or any(self.d.warning.number):fall=True;break
        duration=executed*self.m.opt.timestep;self.elapsed+=duration;self.loads=load_impulse/max(duration,1e-9);self._forward()
        yaw=self._yaw();dyaw=math.atan2(math.sin(yaw-self.last_yaw),math.cos(yaw-self.last_yaw));self.last_yaw=yaw
        supported=bool(sum(self.loads)>.08*self.weight);x,y,z=self.d.qpos[:3];up=float(self.d.xmat[1].reshape(3,3)[2,2])
        self.seen_support|=supported
        if self.phase=='approach':
            self.no_contact_time=0 if supported else self.no_contact_time+duration
            if self.seen_support and x>=21.85 and self.no_contact_time>=.04:
                self.phase='airborne';self.launched=True;self.takeoff_x=float(x);self.airtime=0.;self.spin=0.
        elif self.phase=='airborne':
            self.airtime+=duration;self.total_airtime+=duration;self.spin+=dyaw
            if self.target_turns>0:
                sx=[self.d.xmat[b].reshape(3,3)[:,0] for b in self.skis];angle=math.acos(np.clip(sx[0]@sx[1],-1,1));
                R=self.d.xmat[1].reshape(3,3);axes=[s@R for s in sx];centres=[self.d.xpos[b]@R for b in self.skis];matrix=np.column_stack([axes[0][:2],-axes[1][:2]])
                crossed=False
                if abs(np.linalg.det(matrix))>.015:
                    distance=np.linalg.solve(matrix,centres[1][:2]-centres[0][:2]);crossed=bool(np.all((distance>.12)&(distance<.65)))
                palm,tail=grab_points(self.m,self.d);gap=np.linalg.norm(palm-tail)
                self.cross_time+=duration*float(crossed and gap<.06);self.palm_error_integral+=duration*float(gap)
            self.contact_time=self.contact_time+duration if supported else 0
            if self.contact_time>=.02:
                self.phase='landing';self.touched=True;self.spin_at_touchdown=float(self.spin);self.landing_x=float(x)
        elif self.phase=='landing':
            stable=supported and up>.65 and abs(y)<6.5 and np.linalg.norm(self.d.qvel[3:6])<4 and self.d.subtree_linvel[1,0]>.5
            self.stable_time=self.stable_time+duration if stable else 0.
            # Require sustained support AND actual forward glide, not just one contact frame.
            self.success=bool(self.stable_time>=(.65 if self.stage=='landing' else 1.) and x-self.landing_x>=(2.5 if self.stage=='landing' else 5.))
        outside=not (0<=x<=63 and abs(y)<7.7)
        fall=fall or up<.30 or outside
        self.fallen=bool(fall);self.failure_reason='body_snow_contact' if body_collision else ('out_of_bounds' if outside else ('numerical_warning' if any(self.d.warning.number) else ('tilt' if up<.30 else '')))
        # Landing dominates the return. Air trick bonus is paid only after sustained landing.
        reward=duration*(.6*max(up,0)+.3*np.clip(self.d.subtree_linvel[1,0]/7,0,1)-.015*np.mean(a*a)-.03*np.mean((a-self.previous_action)**2))
        if self.phase=='airborne' and self.target_turns>0:
            before=abs(2*np.pi*self.target_turns-(self.spin-dyaw));after=abs(2*np.pi*self.target_turns-self.spin)
            reward+=.5*np.clip(before-after,-.2,.2)
        if self.phase=='landing':reward+=duration*(4*float(supported)+2*max(up,0))
        if self.success:
            err=(self.spin_at_touchdown-2*np.pi*self.target_turns)
            spin_score=math.exp(-err*err/(math.radians(35)**2)) if self.stage=='full' else 0.
            self.trick_bonus=(25*spin_score+5*min(self.cross_time/.4,1)*spin_score) if self.target_turns>0 else 0.
            reward+=100+self.trick_bonus
        if fall:reward-=150
        self.previous_action=a.copy();self.previous_target=target.copy();self.episode_return+=float(reward)
        terminated=bool(fall or self.success);truncated=bool(self.elapsed>=5 and not terminated)
        if truncated:
            reward-=75;self.episode_return-=75;self.failure_reason='time_limit'
        return self._obs(),float(reward),terminated,truncated,self._info()

    def _info(self):
        return dict(stage=self.stage,target_turns=self.target_turns,frame_index=self.frame_index,phase=self.phase,success=self.success,fallen=self.fallen,failure_reason=self.failure_reason,launched=self.launched,touched=self.touched,airtime_s=self.total_airtime,spin_deg=math.degrees(self.spin_at_touchdown if self.touched else self.spin),stable_landing_s=self.stable_time,x_m=float(self.d.qpos[0]),max_penetration_m=self.max_penetration,max_torque_limit_ratio=self.max_joint_torque_ratio,return_so_far=self.episode_return,physical_grasp=False,full_start=self.stage=='full',elapsed_s=self.elapsed)

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.m,720,1000)
        camera=mujoco.MjvCamera();camera.lookat[:]=self.d.subtree_com[1];camera.distance=3.5;camera.azimuth=-110;camera.elevation=-15;self.renderer.update_scene(self.d,camera=camera);return self.renderer.render()

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
