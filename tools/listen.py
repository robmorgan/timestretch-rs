#!/usr/bin/env python3
"""Build concealed comparisons of current providers from generated or supplied audio.

Run render_corpus.py first, or supply float WAV inputs here. Scores use the fixed
portable evaluator with Portable output as a comparison anchor, not an oracle.
"""
import argparse,csv,json,math,random,subprocess
from pathlib import Path
from audio_files import wav_read,wav_write
from benchmark_backends import executable_name,write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('target/qualification'))
    p.add_argument('--binary',type=Path,default=Path('target/release')/executable_name('timestretch-qa'))
    p.add_argument('inputs',nargs='*',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    folder=a.output/'renders'
    if a.inputs or not (folder/'renders.json').exists():
        from render_corpus import render
        render(a.binary.resolve(),folder,quick=True,inputs=a.inputs)
    renders=json.loads((folder/'renders.json').read_text());rows=[];listen=[];key={};rng=random.Random(20261028)
    for name in sorted({r['condition'] for r in renders}):
        candidates=[r for r in renders if r['condition']==name];anchor=next(r for r in candidates if r['arm']=='rust');current=[]
        for r in candidates:
            score=json.loads(subprocess.check_output([str(a.binary),'score',str(folder/anchor['output']),str(folder/r['output'])],text=True))
            item={'condition':name,'arm':r['arm'],'comparison_anchor':'Portable output; not an independent quality target','quality':score};rows.append(item);current.append(item)
        common=min(1.,.98/max(1e-30,max(r['quality']['peak'] for r in current)))
        gains={}
        for r in current:
            loudness=r['quality'].get('loudness') or {};lufs=loudness.get('integrated_lufs')
            gains[r['arm']]=10**((-23-lufs)/20) if lufs is not None and math.isfinite(lufs) else 1.
        trim=min(1.,.98/max(1e-30,max(r['quality']['peak']*gains[r['arm']] for r in current)))
        blind=a.output/'listen'/f'{len(listen):03d}';blind.mkdir(parents=True,exist_ok=True)
        entry={'condition':name,'directory':str(blind.relative_to(a.output)),'labels':[]};key[name]={}
        for i,r in enumerate(rng.sample(candidates,len(candidates))):
            label=chr(65+i);arm=r['arm'];key[name][label]=arm;entry['labels'].append(label)
            ch,sr,data=wav_read(folder/r['output']);gain=gains[arm]*trim
            for mode,g in [('raw',1.),('common',common),('loudness',gain)]:wav_write(blind/f'{label}-{mode}.wav',ch,sr,(x*g for x in data))
            next(row for row in current if row['arm']==arm)['listening_gains']={'unscaled':1.,'common':common,'loudness':gain,'target_lufs':-23.,'common_loudness_trim':trim}
        listen.append(entry)
    write(a.output/'audio-quality.json',rows);write(a.output/'listening.json',listen);write(a.output/'sealed-key.json',key)
    flat=[{'condition':r['condition'],'arm':r['arm'],'comparison_anchor':r['comparison_anchor'],**{k:v for k,v in r['quality'].items() if not isinstance(v,(dict,list))}} for r in rows]
    with (a.output/'audio-quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in flat))));w.writeheader();w.writerows(flat)
    page=Path(__file__).with_name('backend_listen.html').read_text().replace('<option value="common">','<option value="raw">Unscaled</option><option value="common">')
    (a.output/'listen.html').write_text(page)
if __name__=='__main__':main()
