"""Native MuJoCo renders; no generated athlete/robot imagery or altered model geometry."""
import json,math
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from annotate_720 import model,OUT
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
def font(n):return ImageFont.truetype(FONT,n)

def main():
 m,d=model();frames=json.load(open(OUT/'keyframes.json'))['frames'];air=np.load(OUT/'placed_aerial.npz');thumbnails=[];anim=[]
 def decorate(renderer):
  for side in ['left','right']:
   for i,x,z,col in [(6,.2,.065,[.1,1,.3,1]),(0,-.2,.05,[1,.4,.05,1])]:
    b=m.body(side+'_ski_segment_'+str(i)).id;pos=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([x,0,z]);g=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018]*3),pos,np.eye(3).ravel(),np.array(col));renderer.scene.ngeom+=1
 def view(renderer,az,el,dist=2.6):
  c=mujoco.MjvCamera();c.lookat[:]=d.subtree_com[1];c.lookat[2]-=.08;c.distance=dist;c.azimuth=az;c.elevation=el;renderer.update_scene(d,camera=c);decorate(renderer);return Image.fromarray(renderer.render())
 with mujoco.Renderer(m,height=600,width=720) as renderer:
  for f in frames:
   d.qpos[:]=f['qpos'];d.qvel[:]=f['qvel'];mujoco.mj_forward(m,d);yaw=math.degrees(math.atan2(d.xmat[1].reshape(3,3)[1,0],d.xmat[1].reshape(3,3)[0,0]));left=view(renderer,-115,-12);right=view(renderer,yaw-90,-75)
   im=Image.new('RGB',(1440,770),'#102635');im.paste(left,(0,105));im.paste(right,(720,105));dr=ImageDraw.Draw(im);valid=f['masks']['dynamic_state'];color='#74dfce' if valid else '#ffc46b';status='NATIVE AIR STATE' if valid else 'POSE GOAL / GROUND DYNAMICS UNVERIFIED'
   dr.text((24,14),f"{f['index']:02d}   HELICOPTER 720   |   t = {f['time_s']:+.2f} s   |   yaw = {f['body_yaw_unwrapped_deg']:.1f} deg",font=font(28),fill='white');dr.text((24,55),status+'   /   '+f['phase'].replace('_',' ').upper(),font=font(22),fill=color)
   dr.text((22,718),'WORLD AZIMUTH VIEW',font=font(19),fill='white');dr.text((750,718),'BODY-ALIGNED TOP VIEW',font=font(19),fill='white');dr.text((22,745),'GREEN: ski tips (+X)    ORANGE: tails (-X)    |    Fixed G1 fingers: palm contact target only',font=font(17),fill='#bfccd8');im.save(OUT/f['image']);small=im.copy();small.thumbnail((576,308));anim.append(small)
   tile=Image.new('RGB',(400,350),'#102635');thumb=left.resize((400,333));tile.paste(thumb,(0,17));td=ImageDraw.Draw(tile);td.rectangle((0,0,400,62),fill='#102635');td.text((12,8),f"{f['index']:02d}   {f['time_s']:+.2f}s   {f['body_yaw_unwrapped_deg']:.0f} deg",font=font(21),fill='white');td.text((12,36),'AIR VERIFIED' if valid else 'POSE GOAL ONLY',font=font(17),fill=color);thumbnails.append(tile)
  sheet=Image.new('RGB',(2000,2200),'#102635');sd=ImageDraw.Draw(sheet);sd.text((25,18),'G1 + SKIS  /  FRONT-CROSS HELICOPTER 720',font=font(37),fill='white');sd.text((25,67),'30 keyframes  |  Green tips cross forward of bindings  |  Amber frames are unvalidated ground pose goals',font=font(23),fill='#d0dce6')
  for i,tile in enumerate(thumbnails):sheet.paste(tile,((i%5)*400,100+(i//5)*350))
  sheet.save(OUT/'overview.png');anim[0].save(OUT/'keyframes_preview.gif',save_all=True,append_images=anim[1:],duration=180,loop=0)
  # Keep the failed actual landing visible, separate from requested absorption pose goals.
  fail=np.load(OUT/'landing_probe.npz');d.qpos[:]=fail['qpos'][-1];mujoco.mj_forward(m,d);im=view(renderer,-115,-15,3);dr=ImageDraw.Draw(im);dr.rectangle((0,0,720,70),fill='#5b2227');dr.text((14,12),'FAILED FIXED-POSE LANDING / NOT A TRAINING DEMO',font=font(21),fill='white');dr.text((14,42),'Actual native rollout at t = 1.80 s',font=font(19),fill='white');im.save(OUT/'landing_failure.png')
 # Static reference pose comparison, independent of the moving world camera.
 with mujoco.Renderer(m,height=600,width=720) as renderer:
  d.qpos[:]=json.load(open(OUT/'cross_grab_pose_ik.json'))['qpos'];mujoco.mj_forward(m,d)
  comparison=Image.new('RGB',(1800,670),'#102635');cd=ImageDraw.Draw(comparison)
  cd.text((25,18),'USER REFERENCE  /  G1 FRONT-CROSS ADAPTATION',font=font(33),fill='white')
  images=[Image.open(OUT/'reference.jpeg').convert('RGB'),view(renderer,-115,-12,2.25),view(renderer,-90,-85,2.25)]
  labels=['USER PHOTO: crossed fronts / lifted tail','G1 SIDE: palm to rear ski section','G1 TOP: green tips cross before bindings']
  for i,(im,label) in enumerate(zip(images,labels)):
   im.thumbnail((590,530));comparison.paste(im,(i*600+(600-im.width)//2,85+(530-im.height)//2));cd.text((i*600+12,628),label,font=font(19),fill='white')
  comparison.save(OUT/'reference_comparison.png')
 # Whole-scene aerial path, measured positions and COM samples.
 with mujoco.Renderer(m,height=800,width=1200) as renderer:
  d.qpos[:]=air['qpos'][125];mujoco.mj_forward(m,d);c=mujoco.MjvCamera();c.lookat[:]=np.mean(air['com'],axis=0);c.lookat[2]-=1;c.distance=16;c.azimuth=-90;c.elevation=-12;renderer.update_scene(d,camera=c)
  for i in range(0,len(air['time']),8):
   g=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.035]*3),air['com'][i],np.eye(3).ravel(),np.array([.95,.4,.06,1]));renderer.scene.ngeom+=1
  im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,1200,86),fill='#102635');dr.text((22,15),'TABLETOP STEEP  |  35 deg approach / 18 deg lip',font=font(29),fill='white');dr.text((22,52),'7.053 m/s airborne reset  |  1.30 s flight  |  Orange: measured COM parabola',font=font(24),fill='#d0dce6');im.save(OUT/'trajectory.png')
 print('Rendered 30 keyframes, overview, GIF, trajectory and actual failed landing.')
if __name__=='__main__':main()
