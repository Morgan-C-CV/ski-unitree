"""Fetch an immutable official Menagerie G1 revision and its referenced assets."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]/'robots'/'unitree_g1'

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ski-env-builder'})
    with urllib.request.urlopen(req,timeout=90) as r: return r.read()

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'source.json'
    sha=json.loads(lock.read_text())['commit'] if lock.exists() else json.loads(fetch('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
    base=f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sha}/unitree_g1/'
    xml=fetch(base+'g1.xml'); (ROOT/'g1.xml').write_bytes(xml)
    files=['LICENSE','README.md']+['assets/'+m.attrib['file'] for m in ET.fromstring(xml).find('asset').findall('mesh')]
    def one(name):
        p=ROOT/name;p.parent.mkdir(exist_ok=True)
        if not p.exists(): p.write_bytes(fetch(base+name))
        return name,hashlib.sha256(p.read_bytes()).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: hashes=dict(pool.map(one,files))
    hashes['g1.xml']=hashlib.sha256(xml).hexdigest()
    lock.write_text(json.dumps(dict(repository='https://github.com/google-deepmind/mujoco_menagerie',commit=sha,variant='unitree_g1/g1.xml',sha256=hashes),indent=2)+'\n')
    print(f'Fetched {len(hashes)} G1 files at {sha}')

if __name__=='__main__': main()
