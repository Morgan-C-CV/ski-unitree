"""Mac CPU PPO with checkpointed, evaluation-gated landing-first curriculum."""
import argparse,json,math,os,time,hashlib
from pathlib import Path
import numpy as np
# Avoid oversubscribing BLAS in each environment process.
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv,VecNormalize,sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from rl.env import SkiJumpEnv,PROJECT

CURRICULUM=[('landing',0.),('full',0.),('full',.5),('full',1.),('full',1.5),('full',2.)]

def factory(stage,turns,rank):
    def make():
        e=SkiJumpEnv(stage=stage,target_turns=turns);e.reset(seed=100+rank)
        return Monitor(e,info_keywords=('success','fallen','spin_deg','full_start','max_penetration_m'))
    return make

def write_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def evaluate(model,normalizer,stage,turns,episodes=8,seed_base=10000,record=False):
    e=SkiJumpEnv(stage=stage,target_turns=turns);out=[];trajectory=[]
    for i in range(episodes):
        obs,_=e.reset(seed=seed_base+i,options={'randomize':True});done=False
        while not done:
            action,_=model.predict(normalizer.normalize_obs(obs[None,:]),deterministic=True)
            obs,reward,term,trunc,info=e.step(action[0]);done=term or trunc
            if record and i==0:trajectory.append(dict(time=e.elapsed,qpos=e.d.qpos.copy(),qvel=e.d.qvel.copy(),action=action[0].copy(),phase=e.phase))
        out.append(info)
    e.close();rate=float(np.mean([r['success'] for r in out]));spinerr=float(np.mean([abs(r['spin_deg']-360*turns) for r in out]));full=stage=='full'
    # Promotion requires correct jump and stable glide on held-out episodes, not return alone.
    trick_rate=float(np.mean([r['success'] and (not full or (r['launched'] and r['airtime_s']>.15)) and abs(r['spin_deg']-360*turns)<35 for r in out]))
    result=dict(stage=stage,target_turns=turns,episodes=episodes,seeds=list(range(seed_base,seed_base+episodes)),stable_landing_rate=rate,correct_rotation_and_landing_rate=trick_rate,mean_spin_error_deg=spinerr,episodes_detail=out,passed=bool(rate>=.9 and trick_rate>=.9))
    return result,trajectory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=131072);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--block',type=int,default=4096);parser.add_argument('--eval-episodes',type=int,default=10);parser.add_argument('--run',default='rl/runs/mac_ppo');parser.add_argument('--resume',action='store_true');parser.add_argument('--start-stage',type=int,default=0);args=parser.parse_args()
    torch.set_num_threads(1);run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'latest.zip').exists() and not args.resume:raise SystemExit('Checkpoint exists; use --resume or a new --run path')
    config=dict(vars(args),algorithm='PPO',device='cpu',physics_dt=.00025,policy_dt=.01,frame_index=1,initial_speed=7.05253,seed=41,curriculum=CURRICULUM,environment_sha256=hashlib.sha256((PROJECT/'rl/env.py').read_bytes()).hexdigest(),source_scene_sha256=hashlib.sha256((PROJECT/'assets/env/scenes/g1_tabletop_steep.xml').read_bytes()).hexdigest())
    write_json(run/'config.json',config);started=time.monotonic();history=[];level=args.start_stage;total=0;best=(-1.,-1.);model=None;vec=None;promotions=0
    if args.resume:
        previous=json.loads((run/'status.json').read_text());level=previous['level'];total=previous['total_steps'];history=json.loads((run/'evaluations.json').read_text()) if (run/'evaluations.json').exists() else []
    def make_vec(level):
        stage,turns=CURRICULUM[level];constructors=[factory(stage,turns,i) for i in range(args.workers)]
        raw=SubprocVecEnv(constructors,start_method='spawn') if args.workers>1 else DummyVecEnv(constructors)
        v=VecNormalize.load(str(run/'vecnormalize.pkl'),raw) if (run/'vecnormalize.pkl').exists() else VecNormalize(raw,norm_obs=True,norm_reward=True,clip_obs=10,gamma=math.exp(-.01/2))
        v.training=True;return v
    vec=make_vec(level)
    if args.resume:model=PPO.load(run/'latest.zip',env=vec,device='cpu')
    else:model=PPO('MlpPolicy',vec,learning_rate=3e-4,n_steps=256,batch_size=256,n_epochs=5,gamma=math.exp(-.01/2),gae_lambda=.95,clip_range=.2,ent_coef=.002,vf_coef=.5,max_grad_norm=.5,target_kl=.025,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),log_std_init=-1.4),device='cpu',seed=41,verbose=1)
    budget_end=total+args.steps
    try:
        while total<budget_end:
            block=min(args.block,budget_end-total);stage,turns=CURRICULUM[level]
            write_json(run/'status.json',dict(status='training',level=level,total_steps=total,stage=stage,target_turns=turns,pid=os.getpid(),elapsed_wall_s=time.monotonic()-started,full_skill_verified=False))
            before=model.num_timesteps;model.learn(total_timesteps=block,reset_num_timesteps=False);total+=model.num_timesteps-before
            model.save(run/'latest.zip');vec.save(str(run/'vecnormalize.pkl'))
            write_json(run/'status.json',dict(status='evaluating',level=level,total_steps=total,stage=stage,target_turns=turns,pid=os.getpid(),elapsed_wall_s=time.monotonic()-started,full_skill_verified=False))
            result,trajectory=evaluate(model,vec,stage,turns,args.eval_episodes,record=True);result['total_steps']=total;history.append(result);write_json(run/'evaluations.json',history)
            if trajectory:np.savez_compressed(run/'latest_rollout.npz',**{key:np.array([r[key] for r in trajectory]) for key in trajectory[0]})
            score=(result['stable_landing_rate'],result['correct_rotation_and_landing_rate'])
            if score>best:
                best=score;model.save(run/f'best_stage_{level}.zip');vec.save(str(run/f'best_stage_{level}_vecnormalize.pkl'))
            print(json.dumps({k:v for k,v in result.items() if k!='episodes_detail'}),flush=True)
            # Require two evaluations, with a second independently seeded batch, before promotion.
            if result['passed']:
                confirm,_=evaluate(model,vec,stage,turns,max(20,args.eval_episodes),seed_base=20000);confirm['total_steps']=total;history.append(confirm);write_json(run/'evaluations.json',history)
                if confirm['passed']:
                    if level==len(CURRICULUM)-1:
                        write_json(run/'status.json',dict(status='completed_evaluation_gate',level=level,total_steps=total,full_skill_verified=True,stable_landing_rate=confirm['stable_landing_rate'],correct_rotation_and_landing_rate=confirm['correct_rotation_and_landing_rate']));return
                    level+=1;best=(-1.,-1.);vec.close();vec=make_vec(level);model.set_env(vec,force_reset=True);promotions+=1
        write_json(run/'status.json',dict(status='budget_complete_not_certified',level=level,total_steps=total,elapsed_wall_s=time.monotonic()-started,full_skill_verified=False,last_evaluation={k:v for k,v in history[-1].items() if k!='episodes_detail'} if history else None))
    except BaseException as exc:
        model.save(run/'interrupted.zip');vec.save(str(run/'interrupted_vecnormalize.pkl'));write_json(run/'interrupted.json',dict(error=repr(exc),level=level,total_steps=model.num_timesteps));raise
    finally:
        if vec is not None:vec.close()

if __name__=='__main__':main()
