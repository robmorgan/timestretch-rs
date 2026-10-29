#!/usr/bin/env python3
"""Render deterministic signals or supplied float WAV files with current providers.

An optional --compare directory checks same-condition output hashes against a
previous run. It is a regression comparison, not an independent quality oracle.
"""
import argparse, itertools, json, math, platform, shutil, subprocess
from pathlib import Path
from benchmark_backends import digest, executable_name, synthetic, write
from audio_files import wav_read
KINDS = ['impulses','tone','multitone','noise','silence','stereo','antiphase']

def render(binary, out, quick=False, inputs=(), compare=None):
    out.mkdir(parents=True, exist_ok=True)
    request={'binary_sha256':digest(binary),'quick':quick,'inputs':[digest(p) for p in inputs]}
    environment=out/'environment.json'
    if environment.exists():
        existing=json.loads(environment.read_text())
        if any(existing.get(k)!=v for k,v in request.items()):
            raise ValueError('Render directory belongs to a different binary or input set; choose a fresh directory.')
    if (out/'renders.json').exists():
        rows=json.loads((out/'renders.json').read_text())
        for row in rows:
            assert digest(out/row['output'])==row['sha256'],'cached output changed'
            assert digest(out/row['input'])==row['input_sha256'],'cached input changed'
        if compare:
            previous={(r['condition'],r['arm']):r for r in json.loads((compare/'renders.json').read_text())}
            assert len(previous)==len(rows),'comparison condition sets differ'
            assert all(previous[(r['condition'],r['arm'])]['sha256']==r['sha256'] and previous[(r['condition'],r['arm'])]['input_sha256']==r['input_sha256'] for r in rows),'output changed'
        return rows
    arms=['rust','fft','math','apple'] if platform.system()=='Darwin' else ['rust']
    sources=[]
    configs=itertools.product([48000] if quick else [44100,48000,96000], [1,2], ['tone'] if quick else KINDS)
    for sr,ch,kind in configs:
        name=f'{sr}-{ch}-{kind}';path=out/(name+'.wav');synthetic(path,sr,kind,ch);sources.append((name,path))
    for i,path in enumerate(inputs):
        name=f'input-{i:03d}';copy=out/(name+'.wav');shutil.copy2(path,copy);sources.append((name,copy))
    old={(r['condition'],r['arm']):r for r in json.loads((compare/'renders.json').read_text())} if compare else None
    rows=[]
    write(out/'environment.json',{**request,'platform':platform.platform(),'scope':'generated signals and explicitly supplied float WAV inputs; no external oracle'})
    for name,source in sources:
        for duration in [.75,1.,1.5]:
            condition=f'{name}-{duration:g}';folder=out/condition;folder.mkdir(exist_ok=True)
            for arm in arms:
                target=folder/(arm+'.wav')
                meta=json.loads(subprocess.check_output([str(binary),'render',arm,str(source),str(target),str(duration)],text=True))
                ch,sr,samples=wav_read(target)
                assert samples and all(math.isfinite(x) for x in samples),(condition,arm)
                row={'condition':condition,'arm':arm,'input':str(source.relative_to(out)),'input_sha256':digest(source),'output':str(target.relative_to(out)),'sha256':digest(target),'metadata':meta}
                if old is not None:
                    previous=old[(condition,arm)]
                    row['matches_previous']=row['input_sha256']==previous['input_sha256'] and row['sha256']==previous['sha256']
                    assert row['matches_previous'],(condition,arm,'output changed')
                rows.append(row)
            print(condition,flush=True)
    if old is not None:assert len(rows)==len(old),'comparison condition sets differ'
    write(out/'renders.json',rows)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,default=Path('target/release')/executable_name('timestretch-qa'));p.add_argument('--output',type=Path,default=Path('target/qualification/renders'));p.add_argument('--quick',action='store_true');p.add_argument('--compare',type=Path);p.add_argument('inputs',nargs='*',type=Path);a=p.parse_args()
    render(a.binary.resolve(),a.output,a.quick,a.inputs,a.compare)
if __name__=='__main__':main()
