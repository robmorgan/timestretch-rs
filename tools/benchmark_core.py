#!/usr/bin/env python3
"""Preallocated core timing for the available numerical providers. Complete host timing is separate.

python3 tools/benchmark_core.py --output target/qualification/core
Use --focused for only 96 kHz stereo/64/2x. Initialization is separate from processing.
"""
import argparse,itertools,json,math,os,platform,random,shutil,struct,subprocess
from pathlib import Path
from benchmark_backends import ROOT,digest,executable_name,metadata,perf_row,read_result,run,save_performance,write

def fixture(path,sr,ch):
    f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    data=[]
    for i in range(sr):
        t=i/sr
        v=f32(.2*math.sin(math.tau*220*t)+.1*math.sin(math.tau*880*t)+.4*math.exp(-40*(t%.5))*math.sin(math.tau*60*t))
        data.extend(v*f32(1-.2*c) for c in range(ch))
    path.write_bytes(struct.pack('<'+'f'*len(data),*data))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'target/qualification/core')
    p.add_argument('--binary',type=Path);p.add_argument('--focused',action='store_true');p.add_argument('--no-build',action='store_true');a=p.parse_args();os.chdir(ROOT)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'performance.json').exists():raise SystemExit('Use a fresh output directory to preserve measurements.')
    if not a.no_build and not a.binary:run(['cargo','build','--release','-p','timestretch-qa'])
    binary=out/executable_name('timestretch-qa');shutil.copy2(a.binary or ROOT/'target/release'/executable_name('timestretch-qa'),binary)
    apple=platform.system()=='Darwin';arms=['rust','fft','math','apple'] if apple else ['rust']
    write(out/'environment.json',metadata())
    write(out/'binary-hashes.json',{binary.name:digest(binary)})
    configs=[(96000,2,64,2)] if a.focused else list(itertools.product([44100,48000,96000],[1,2],[64,256,1024],[.5,1,2]))
    for sr,ch in sorted({c[:2] for c in configs}):fixture(out/f'{sr}-{ch}.raw',sr,ch)
    write(out/'fixtures.json',{x.name:digest(x) for x in out.glob('*.raw')})
    def command(arm,cfg,n,audit=False):
        sr,ch,cb,tempo=cfg
        return [binary,'core',arm]+[sr,ch,cb,tempo,n,out/f'{sr}-{ch}.raw']+(['audit'] if audit else [])
    audits=[]
    keys=['calls','consumed','pending','prefill_input','prefill_output','actual_stretch','schedule_hash']
    for cfg in configs:
        pair=[]
        for arm in arms:
            row=read_result(command(arm,cfg,1000,True));row.pop('times_ns');pair.append(row)
        assert all(all(row[k]==pair[0][k] for k in keys) for row in pair),(cfg,pair)
        audits.extend(pair)
    write(out/'schedule-audit.json',audits)
    rows=[]
    def measure(jobs):
        random.Random(20261028).shuffle(jobs)
        for arm,cfg,n,rep,phase in jobs:
            print(phase,arm,cfg,rep,flush=True)
            row=perf_row(read_result(command(arm,cfg,n)));row.update(arm=arm,repetition=rep,phase=phase)
            assert row.get('allocations',0)==0
            rows.append(row)
            with (out/'performance.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        save_performance(out,rows)
    measure([(arm,cfg,1000,rep,'screen') for cfg in configs for arm in arms for rep in range(3)])
    jobs=[]
    for arm in arms:
        screen=[r for r in rows if r['arm']==arm]
        worst={tuple(max(screen,key=lambda r:r[metric]/r['budget_ns'])[k] for k in ['sample_rate','channels','callback','tempo']) for metric in ['max_ns','p99_ns']}
        worst.add((96000,2,64,2))
        jobs.extend((arm,cfg,10000,rep,'tail') for cfg in worst for rep in range(5))
    measure(jobs)
    if apple:
        probe=out/'allocations.dylib';run(['clang','-O2','-dynamiclib',ROOT/'tools/backend_allocations.c','-o',probe])
        env=dict(os.environ,DYLD_INSERT_LIBRARIES=str(probe));positive=json.loads(subprocess.check_output([binary,'audit-probe'],env=env));assert positive['system_allocations']>0
        allocations=[]
        for arm in arms:
            row=json.loads(subprocess.check_output([str(x) for x in command(arm,(96000,2,64,2),1000)],env=env));row.pop('times_ns');allocations.append(row)
            assert row['system_allocations']==0
        write(out/'allocations.json',{'positive_control':positive,'workloads':allocations})
    print(out,flush=True)
if __name__=='__main__':main()
