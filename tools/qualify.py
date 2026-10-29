#!/usr/bin/env python3
"""timestretch qualification; resumable raw measurements and release gates.

python3 tools/qualify.py --stage playback
The full wall-clock playback protocol takes about 45 minutes. Run on an idle
machine for a baseline; record other workloads separately. This is CPU/queue
qualification; physical-device soaks are recorded separately. No automated
measurement constitutes a human listening verdict. Requires Python 3.11+.
"""
import argparse,csv,hashlib,html,itertools,json,os,platform,random,shutil,statistics,subprocess,sys,time,tomllib
from pathlib import Path
from benchmark_backends import executable_name
ROOT=Path(__file__).resolve().parents[1]
def run(args,**kw): return subprocess.run([str(x) for x in args],check=True,**kw)
def capture(args): return subprocess.check_output([str(x) for x in args],text=True).strip()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(binary,out,kind,key,args):
    folder=out/kind;folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    print(kind,key,flush=True);start=time.time()
    row=json.loads(capture([binary,*args]));row['run_id']=key;row['elapsed_seconds']=time.time()-start
    write(path,row);return row

def playback(binary,out,quick):
    conditions=list(itertools.product(['rust','apple'] if platform.system()=='Darwin' else ['rust'],[44100,48000,96000],[1,2],[64,256,1024],[0.5,1,2]))
    if quick:conditions=[(arm,96000,2,64,2) for arm in ['rust','apple'] if arm=='rust' or platform.system()=='Darwin']
    jobs=[(c,r) for c in conditions for r in range(1 if quick else 3)];random.Random(20261028).shuffle(jobs)
    rows=[]
    for (arm,sr,ch,cb,tempo),rep in jobs:
        key=f'screen-{arm}-{sr}-{ch}-{cb}-{tempo}-{rep}'
        rows.append(measure(binary,out,'playback',key,['playback',arm,sr,ch,cb,tempo,1000]))
    for arm in sorted({c[0] for c in conditions}):
        r=[x for x in rows if x['arm']==arm]
        worst={tuple(max(r,key=lambda x:x[metric]/x['budget_ns'])[k] for k in ['sample_rate','channels','callback','tempo']) for metric in ['max_ns','p99_ns']}
        worst.add((96000,2,64,2))
        for sr,ch,cb,tempo in sorted(worst):
            for rep in range(1 if quick else 5):
                rows.append(measure(binary,out,'playback',f'tail-{arm}-{sr}-{ch}-{cb}-{tempo}-{rep}', ['playback',arm,sr,ch,cb,tempo,10000]))
        for reserve,stress in itertools.product([1024,2048,4096,4800,8192],[False,True]):
            rows.append(measure(binary,out,'playback',f'reserve-{arm}-{reserve}-{stress}', ['playback',arm,96000,2,64,2,1000 if quick else 10000,reserve]+(['stress'] if stress else [])))
    return rows

def load(out,name,default=None):
    path=out/name
    return json.loads(path.read_text()) if path.exists() else default

def failed_playback(r):
    return bool(r['underruns'] or r.get('queue_underruns_including_warmup') or r['deadline_misses'] or r['allocations'] or r.get('deallocations') or r.get('system_allocations') not in (None,0) or r['worker_error'] or r.get('worker_qos_error') or r.get('worker_scheduling_error') or r.get('callback_scheduling_error') or not r['finite'] or (r['p999_ns'] is not None and r['p999_ns']>0.5*r['budget_ns']))

def complete_protocol(rows, arms):
    fields=['sample_rate','channels','callback','tempo']
    config=lambda r:tuple(r[k] for k in fields)
    expected=set(itertools.product([44100,48000,96000],[1,2],[64,256,1024],[.5,1,2]))
    for arm in arms:
        screen=[r for r in rows if r['arm']==arm and r['run_id'].startswith('screen-') and r['callbacks']==1000]
        if any(sum(config(r)==c for r in screen)<3 for c in expected):return False
        worst={config(max(screen,key=lambda r:r[k]/r['budget_ns'])) for k in ['max_ns','p99_ns']}
        worst.add((96000,2,64,2))
        tails=[r for r in rows if r['arm']==arm and r['run_id'].startswith('tail-') and r['callbacks']>=10000]
        if any(sum(config(r)==c for r in tails)<5 for c in worst):return False
    return bool(arms)

def device(binary,out,seconds):
    if seconds==0:return
    jobs=list(itertools.product(['rust','apple'] if platform.system()=='Darwin' else ['rust'],[44100,48000,96000],[64,256,1024]))
    random.Random(20261028).shuffle(jobs)
    for arm,sr,cb in jobs:
        key=f'{arm}-{sr}-{cb}';path=out/'device'/(key+'.json');path.parent.mkdir(exist_ok=True)
        if path.exists():continue
        try:measure(binary,out,'device',key,['device',arm,sr,cb,seconds])
        except subprocess.CalledProcessError as error:write(path,{'run_id':key,'error':str(error),'qualified':False})

def boundary_playback(binary,out,quick):
    jobs=list(itertools.product(['rust','apple'] if platform.system()=='Darwin' else ['rust'],[44100,48000,96000],[1,2]))
    random.Random(20261028).shuffle(jobs)
    for arm,sr,ch in jobs:
        measure(binary,out,'boundary',f'boundary-{arm}-{sr}-{ch}',
                ['playback',arm,sr,ch,64,1,1000 if quick else 10000,(sr+19)//20,'boundary'])

def complete_device_protocol(rows,arms):
    expected=set(itertools.product(arms,[44100,48000,96000],[64,256,1024]))
    actual={(r['run_id'].split('-')[0],r.get('source_rate'),r.get('requested_callback')) for r in rows if r.get('seconds',0)>=15 and r.get('callbacks',0)>0}
    return bool(arms) and expected <= actual

def complete_boundary_protocol(rows,arms):
    expected=set(itertools.product(arms,[44100,48000,96000],[1,2]))
    required_ratios={(1/32,1.0),(1/32,0.25),(1/16,2.0),(1/8,4.0),(0.25,0.25),(0.25,1.0),(0.25,4.0),(1.0,0.25),(1.0,4.0),(4.0,1.0),(4.0,4.0),(1.0,1.0)}
    actual={(r['arm'],r['sample_rate'],r['channels']) for r in rows if r.get('boundary_stress') and r['callbacks']>=10000 and required_ratios <= {tuple(pair) for pair in r.get('boundary_tempo_pitch',[])}}
    return bool(arms) and expected <= actual

def audits(binary,out):
    env=dict(os.environ)
    if platform.system()=='Darwin':
        probe=out/'allocation_probe.dylib';run(['clang','-O2','-dynamiclib',ROOT/'tools/backend_allocations.c','-o',probe]);env['DYLD_INSERT_LIBRARIES']=str(probe)
    def audited(args):return json.loads(subprocess.check_output([str(binary),*map(str,args)],text=True,env=env))
    control=audited(['audit-probe']);assert control['host_allocations']>0
    if platform.system()=='Darwin':assert control['system_allocations']>0
    write(out/'allocator-control.json',control);write(out/'callback-allocations.json',audited(['callback-audit']))
    rows=[]
    for arm in (['rust','apple'] if platform.system()=='Darwin' else ['rust']):
        row=audited(['session',arm,96000,2,64,2,10000,'stress']);row['arm']=arm
        assert row['allocations']==0 and row['deallocations']==0 and row['finite']
        if platform.system()=='Darwin':assert row['system_allocations']==0
        rows.append(row)
    write(out/'session-allocations.json',rows)

def session_performance(binary,out,quick):
    from benchmark_backends import perf_row,save_performance
    folder=out/'session';folder.mkdir(exist_ok=True)
    if (folder/'performance.json').exists():return
    configs=[(96000,2,64,2)] if quick else list(itertools.product([44100,48000,96000],[1,2],[64,256,1024],[.5,1,2]))
    arms=['rust','apple'] if platform.system()=='Darwin' else ['rust'];rows=[]
    jobs=list(itertools.product(arms,configs,range(1 if quick else 3)));random.Random(20261028).shuffle(jobs)
    for arm,cfg,rep in jobs:
        key=f'screen-{arm}-'+ '-'.join(map(str,cfg))+f'-{rep}'
        row=perf_row(measure(binary,out,'session',key,['session',arm,*cfg,1000]));row.update(arm=arm,phase='screen',repetition=rep);rows.append(row)
    for arm in arms:
        screen=[r for r in rows if r['arm']==arm]
        worst={tuple(max(screen,key=lambda r:r[metric]/r['budget_ns'])[k] for k in ['sample_rate','channels','callback','tempo']) for metric in ['max_ns','p99_ns']};worst.add((96000,2,64,2))
        for cfg,rep in itertools.product(sorted(worst),range(1 if quick else 5)):
            key=f'tail-{arm}-'+ '-'.join(map(str,cfg))+f'-{rep}'
            row=perf_row(measure(binary,out,'session',key,['session',arm,*cfg,10000]));row.update(arm=arm,phase='tail',repetition=rep);rows.append(row)
    save_performance(folder,rows)

def summarize(out, measurements=None, playback_results=None, device_results=None, listening_results=None, audit_results=None, boundary_results=None):
    measurements=measurements or out; playback_results=playback_results or out
    device_results=device_results or out; listening_results=listening_results or out
    audit_results=audit_results or measurements; boundary_results=boundary_results or playback_results
    folders={'processing':measurements,'host':playback_results,'device':device_results,'allocations':audit_results,'boundary':boundary_results,'listening':listening_results}
    sources={k:{'directory':os.path.relpath(v,out),'environment':load(v,'environment.json',{})} for k,v in folders.items()}
    rows=[json.loads(p.read_text()) for p in sorted((playback_results/'playback').glob('*.json'))]
    dev=[json.loads(p.read_text()) for p in sorted((device_results/'device').glob('*.json'))]
    boundary=[json.loads(p.read_text()) for p in sorted((boundary_results/'boundary').glob('*.json'))]
    quality=load(measurements,'quality.json',[])
    callback=load(audit_results,'callback-allocations.json',[]);allocation=load(audit_results,'session-allocations.json',[])
    host_env=sources['host']['environment'];recorded_platform=host_env.get('platform',platform.system())
    arms=['rust','apple'] if recorded_platform.lower().startswith(('darwin','macos')) else ['rust']
    required=[r for r in rows if not r['run_id'].startswith('reserve-') or r['reserve_frames']==(r['sample_rate']*host_env.get('desktop_reserve_ms',50)+999)//1000]
    host_failed=[r['run_id'] for r in rows if failed_playback(r)]
    device_failed=[r['run_id'] for r in dev if r.get('error') or r.get('worker_error') or r.get('worker_scheduling_error') or r.get('device_errors') or r.get('queue_underruns') or r.get('deadline_misses') or (r.get('p999_budget_ratio') is not None and r['p999_budget_ratio']>0.5)]
    allocated=lambda data:bool(data) and all(r['allocations']==0 and r['deallocations']==0 and r['finite'] and r.get('system_allocations') in (None,0) for r in data)
    gates={'checked_in_fixtures':load(measurements,'fixtures.json',{}).get('passed',False),
        'full_host_protocol':complete_protocol(rows,arms),
        'host_timing_and_starvation':bool(required) and not any(failed_playback(r) for r in required),
        'selected_policy_stress':all(any(r['arm']==arm and r.get('stress') and r['callbacks']>=10000 for r in required) for arm in arms),
        'boundary_stress':complete_boundary_protocol(boundary,arms) and not any(failed_playback(r) for r in boundary),
        'retained_quality':bool(quality) and all(r['pass'] for r in quality),
        'callback_allocations':allocated(callback),'session_allocations':allocated(allocation)}
    device_gates={'full_device_protocol':complete_device_protocol(dev,arms),'device_timing_and_starvation':bool(dev) and not device_failed}
    core=load(measurements,'core/performance.json',[]);session=load(measurements,'session/performance.json',[])
    report={'evidence_sources':sources,'local_gates':gates,'device_gates':device_gates,'quality':quality,
        'playback_runs':len(rows),'playback_failed_runs':host_failed,'required_playback_runs':len(required),
        'device_failed_runs':device_failed,'device_runs':len(dev),'core_runs':len(core),'session_runs':len(session),
        'rendered_conditions':len(load(measurements,'renders/renders.json',[])),
        'device_status':'not measured' if not dev else 'passed' if all(device_gates.values()) else 'incomplete or failed',
        'qualification':'finite local processing/host gates passed' if all(gates.values()) else 'incomplete or failed',
        'human_listening':'no preference recorded; audition files are optional',
        'scope':'Current providers, checked-in fixtures and generated signals. Historical external equivalence is not evaluated.'}
    write(out/'summary.json',report)
    def scalar(data):return [{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in data]
    tables={'playback':scalar(rows),'boundary':scalar(boundary),'device':scalar(dev),'quality':scalar(quality),'core':scalar(core),'session':scalar(session)}
    esc=lambda s:html.escape(str(s))
    body='<h1>timestretch qualification</h1><p>'+esc(report['qualification'])+'. Device: '+esc(report['device_status'])+'.</p><p>Physical deadlines and the 50% p99.9 headroom target are unchanged. Quick runs cannot qualify the full matrix. Missing device evidence is never a device pass.</p>'
    body+='<p>'+esc(report['scope'])+'</p><p><a href="summary.json">JSON, gates and provenance</a></p>'
    for name,data in tables.items():
        if not data:continue
        keys=sorted(set().union(*(r.keys() for r in data)))
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
        body+='<h2>'+esc(name)+'</h2><a href="'+name+'.csv">CSV</a>'
        # Full scalar fields remain in CSV; render the metrics most useful for review.
        columns=[k for k in ['run_id','arm','backend','metric','value','bound','limit','pass','sample_rate','channels','callback','tempo','reserve_ms','median_ns','p95_ns','p99_ns','p999_ns','max_ns','deadline_misses','underruns','queue_underruns','allocations','worker_max_ns'] if k in keys]
        body+='<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(r.get(k,''))+'</td>' for k in columns)+'</tr>' for r in data)+'</table>'
    body+='<h2>Additional evidence</h2>'
    for folder,path,label in [(measurements,'fixtures.json','Fixture tests'),(measurements,'micro.json','FFT/math microbenchmarks'),(measurements,'renders/renders.json','Render hashes'),(audit_results,'callback-allocations.json','Callback audit'),(audit_results,'session-allocations.json','Session audit'),(listening_results,'listen.html','Concealed listening'),(listening_results,'audio-quality.csv','Listening measurements')]:
        if (folder/path).exists():body+='<p><a href="'+esc(os.path.relpath(folder/path,out))+'">'+label+'</a></p>'
    body+='<p>Core and session costs are unpaced DSP measurements, separate from delivery callbacks. Opaque FFT-plan memory is unknown. Listening scores compare current providers, not an independent ideal output. Raw samples are never aligned or gain-corrected for scoring.</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>timestretch qualification</title><style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem;border-bottom:1px solid #ddd}h2{margin-top:2rem}</style>'+body)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','fixtures','renders','playback','boundary','micro','quality','core','session','audit','device','listening','report'],default='all')
    p.add_argument('--output',type=Path,default=Path('target/qualification'))
    p.add_argument('--quick',action='store_true');p.add_argument('--no-build',action='store_true')
    p.add_argument('--device-seconds',type=int,default=0,help='Opt-in muted physical device testing; use at least 15 seconds per condition')
    p.add_argument('--strict',action='store_true',help='Exit nonzero for missing or failed local timing, starvation, allocation or quality gates')
    for option in ['measurements','playback-results','device-results','listening-results','audit-results','boundary-results']:
        p.add_argument('--'+option,type=Path,help='Explicit report evidence directory; each revision keeps its own provenance')
    a=p.parse_args();os.chdir(ROOT);out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    binary=out/executable_name('qualification-host')
    if a.stage!='report' and not binary.exists():
        if not a.no_build:run(['cargo','build','--release','-p','timestretch-qa'])
        shutil.copy2(ROOT/'target/release'/executable_name('timestretch-qa'),binary)
    if a.stage!='report' and not (out/'environment.json').exists():
        files=capture(['git','ls-files','--cached','--others','--exclude-standard']).splitlines()
        write(out/'environment.json',{'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dependency_versions':{p['name']:p['version'] for p in tomllib.loads((ROOT/'Cargo.lock').read_text())['package'] if p['name'] in ['rustfft','cpal','audio_thread_priority','timestretch-core','timestretch-accelerate','timestretch-analysis']},'os_build':capture(['sw_vers','-buildVersion']) if platform.system()=='Darwin' else platform.release(),'sdk_version':capture(['xcrun','--show-sdk-version']) if platform.system()=='Darwin' else None,'platform':platform.platform(),'hardware':capture(['sysctl','-n','machdep.cpu.brand_string']) if platform.system()=='Darwin' else platform.machine(),'compiler':capture(['rustc','-Vv']),'binary_sha256':digest(binary),'source_sha256':{f:digest(Path(f)) for f in files if Path(f).is_file()},'profile':'release opt-level=3 lto=true','rustflags':os.environ.get('RUSTFLAGS',''),'seed':20261028,'quick':a.quick,'workload':'developer workstation; host load is not isolated','desktop_reserve_ms':50,'worker_stage_quota':8,'worker_sleep_ms':1,'worker_scheduling':'Mach audio time constraint' if platform.system()=='Darwin' else 'platform default','simulated_callback_scheduling':'Mach audio time constraint' if platform.system()=='Darwin' else 'platform default','pump_policy':'drain completed session output before advancing; exact reserve limit'})
    if a.stage!='report':
        recorded=load(out,'environment.json',{})
        if recorded.get('binary_sha256')!=digest(binary):raise SystemExit('Pinned executable changed; use a fresh output directory.')
        for name,sha in recorded.get('source_sha256',{}).items():
            if name.startswith(('src/','crates/','desktop/src/','tools/')) and (not Path(name).is_file() or digest(Path(name))!=sha):
                raise SystemExit('Measurement sources changed; use a fresh output directory.')
    if a.stage in ['all','fixtures']:
        write(out/'fixtures.json',{'passed':False})
        with (out/'fixtures.log').open('w') as log:
            run(['cargo','test','--release','-p','timestretch-core','--features','qa,accelerate'],stdout=log,stderr=subprocess.STDOUT)
        write(out/'fixtures.json',{'passed':True,'scope':'checked-in component and scheduler fixtures; documented provider rounding bounds'})
    if a.stage in ['all','renders']:
        from render_corpus import render
        render(binary,out/'renders',a.quick)
    if a.stage in ['all','micro']:write(out/'micro.json',{arm:json.loads(capture([binary,'micro',arm])) for arm in (['rust','fft','math','apple'] if platform.system()=='Darwin' else ['rust'])})
    if a.stage in ['all','quality']:write(out/'quality.json',json.loads(capture([binary,'quality'])))
    if a.stage in ['all','core']:run([sys.executable,'tools/benchmark_core.py','--binary',binary,'--output',out/'core']+(['--focused'] if a.quick else []))
    if a.stage in ['all','session']:session_performance(binary,out,a.quick)
    if a.stage in ['all','audit']:audits(binary,out)
    if a.stage in ['all','playback']:playback(binary,out,a.quick)
    if a.stage in ['all','boundary']:boundary_playback(binary,out,a.quick)
    if a.stage in ['all','device']:device(binary,out,a.device_seconds)
    if a.stage in ['all','listening'] and (out/'renders/renders.json').exists():run([sys.executable,'tools/listen.py','--binary',binary,'--output',out])
    report=summarize(out,a.measurements,a.playback_results,a.device_results,a.listening_results,a.audit_results,a.boundary_results);print(json.dumps({k:v for k,v in report.items() if k not in ['quality','device_soak','callback_audit','evidence_sources','boundary_stress_runs','rejected_policies']},indent=2))
    if a.strict and (not all(report['local_gates'].values()) or (a.device_seconds>0 and not all(report['device_gates'].values()))):raise SystemExit(1)
if __name__=='__main__':main()
