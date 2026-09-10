"""Recheck native airborne time-step convergence with the selected shooting solution."""
import json
from annotate_720 import OUT
from flight_720 import flight
if __name__=='__main__':
 r=json.loads((OUT/'aerial_shooting_probe.json').read_text())
 _,fine=flight(r['Lz'],r['T'],.000125,save=False,tilt=r['tilt'])
 (OUT/'aerial_dt_convergence.json').write_text(json.dumps(fine,indent=2)+'\n')
 flight(r['Lz'],r['T'],.00025,tilt=r['tilt'])
