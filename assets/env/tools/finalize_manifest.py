"""Refresh the deliverable hashes after assembly and validation (no self-hashing)."""
import hashlib
import json
from build_assets import ROOT

def main():
    path=ROOT/'manifest.json';manifest=json.loads(path.read_text())
    manifest['files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['scenes','meshes','terrain','configs','tools'] for p in sorted((ROOT/directory).glob('*')) if p.is_file()}
    manifest['official_robot_source']='robots/unitree_g1/source.json'
    manifest['validation_report']='reports/validation.json'
    manifest['exploratory_course_report']='reports/course_probes.json'
    if (ROOT/'reports/dynamics_audit/verdict.json').exists():
        manifest['dynamics_acceptance_report']='reports/dynamics_audit/verdict.json'
        manifest['dynamics_acceptance_status']=json.loads((ROOT/'reports/dynamics_audit/verdict.json').read_text())['overall']
    if (ROOT/'reports/repairs_v1_2/verdict.json').exists():
        manifest['dynamics_acceptance_report']='reports/repairs_v1_2/verdict.json'
        manifest['dynamics_acceptance_status']=json.loads((ROOT/'reports/repairs_v1_2/verdict.json').read_text())['overall']
    manifest['training_ready']=False
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Finalized {len(manifest["files"])} file hashes')

if __name__=='__main__':main()
