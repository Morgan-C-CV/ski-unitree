"""Shoot initial airborne angular momentum/orientation, retaining native actuation."""
import json
import numpy as np
from scipy.optimize import least_squares
from flight_720 import flight
from annotate_720 import OUT
history=[]
def residual(x):
 _,r=flight(float(x[2]),1.3,.0005,save=False,tilt=x[:2]);history.append(r)
 (OUT/'shooting_history.json').write_text(json.dumps(history,indent=2)+'\n')
 roll,pitch,_=np.radians(r['final_rpy_deg']);spin=np.radians(r['spin_deg']-720)
 return np.array([roll,pitch,spin, max(0,.65-r['min_up'])*3])
best=None
for initial in [[.5,-.1,14.5],[-.1,-.2,14.5]]:
 r=least_squares(residual,initial,bounds=([-.9,-.9,10],[.9,.9,22]),diff_step=.003,max_nfev=20,ftol=1e-5,xtol=1e-5,gtol=1e-5)
 if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun):best=r
 if np.linalg.norm(r.fun)<.05:break
x=best.x;rows,report=flight(float(x[2]),1.3,.00025,tilt=x[:2]);report['shoot_parameters']=x.tolist();report['shoot_residual']=best.fun.tolist();(OUT/'aerial_shooting_probe.json').write_text(json.dumps(report,indent=2)+'\n')
