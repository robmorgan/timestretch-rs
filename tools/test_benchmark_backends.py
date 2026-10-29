"""Protocol checks independent of music, macOS and DSP providers."""

import json
import itertools
from pathlib import Path
import tempfile
import unittest
from benchmark_backends import first_difference, perf_row, synthetic
from audio_files import wav_read
from qualify import complete_boundary_protocol, complete_device_protocol, complete_protocol, failed_playback


class Protocol(unittest.TestCase):
    def test_windows_qualification_copies_executable_and_reuses_python(self):
        from contextlib import redirect_stdout
        import io
        import sys
        from unittest.mock import patch
        import qualify

        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();out=root/'qualification';out.mkdir()
            source=root/'target/release/timestretch-qa.exe'
            source.parent.mkdir(parents=True);source.write_bytes(b'qualification host')
            (out/'environment.json').write_text(json.dumps({
                'binary_sha256':qualify.digest(source),'source_sha256':{}}))
            with (patch.object(qualify,'ROOT',root),
                  patch.object(qualify.os,'chdir'),
                  patch.object(qualify.platform,'system',return_value='Windows'),
                  patch.object(sys,'argv',['qualify.py','--stage','core','--no-build','--output',str(out)]),
                  patch.object(qualify,'run') as run,
                  patch.object(qualify,'summarize',return_value={}),
                  redirect_stdout(io.StringIO())):
                qualify.main()
            binary=out/'qualification-host.exe'
            self.assertEqual(binary.read_bytes(),source.read_bytes())
            run.assert_called_once_with([
                sys.executable,'tools/benchmark_core.py','--binary',binary,
                '--output',out/'core'])

    def test_empty_report_does_not_pass_or_link_missing_evidence(self):
        from qualify import summarize
        import re
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            report=summarize(root)
            self.assertFalse(any(report['local_gates'].values()))
            self.assertEqual(report['device_status'],'not measured')
            for target in re.findall(r'href="([^"]+)"',(root/'index.html').read_text()):
                self.assertTrue((root/target).exists(),target)

    def test_cached_render_rejects_modified_audio(self):
        from benchmark_backends import digest
        from render_corpus import render
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'host';binary.write_bytes(b'not executed')
            source=root/'input.wav';source.write_bytes(b'original')
            output=root/'output.wav';output.write_bytes(b'output')
            (root/'environment.json').write_text(json.dumps(dict(binary_sha256=digest(binary),quick=True,inputs=[])))
            (root/'renders.json').write_text(json.dumps([dict(condition='tone',arm='rust',input='input.wav',output='output.wav',input_sha256=digest(source),sha256=digest(output))]))
            output.write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'cached output changed'):
                render(binary,root,quick=True)

    def test_fast_empty_callback_is_still_a_failure(self):
        row=dict(underruns=1,deadline_misses=0,allocations=0,worker_error=None,finite=True,p999_ns=1,budget_ns=1000)
        self.assertTrue(failed_playback(row))
        row['underruns']=0
        self.assertFalse(failed_playback(row))
        row['worker_qos_error']=1
        self.assertTrue(failed_playback(row))
        row['worker_qos_error']=None
        row['queue_underruns_including_warmup']=1
        self.assertTrue(failed_playback(row))
        row['queue_underruns_including_warmup']=0
        row['system_allocations']=1
        self.assertTrue(failed_playback(row))

    def test_duplicate_or_partial_matrix_cannot_qualify(self):
        row=dict(arm='rust',run_id='screen-rust',callbacks=1000,sample_rate=96000,channels=2,callback=64,tempo=2,max_ns=1,p99_ns=1,budget_ns=100)
        self.assertFalse(complete_protocol([row]*324,['rust']))
        self.assertFalse(complete_protocol([],['rust','apple']))

    def test_device_requires_every_condition_and_measured_duration(self):
        rows=[dict(run_id=f'{arm}-{sr}-{cb}',source_rate=sr,requested_callback=cb,seconds=15,callbacks=100)
              for arm,sr,cb in itertools.product(['rust','apple'],[44100,48000,96000],[64,256,1024])]
        self.assertTrue(complete_device_protocol(rows,['rust','apple']))
        self.assertFalse(complete_device_protocol(rows[:-1],['rust','apple']))
        rows[-1]['seconds']=1
        self.assertFalse(complete_device_protocol(rows,['rust','apple']))

    def test_boundary_requires_real_boundary_work_and_full_sample_count(self):
        ratios=[(1/32,1),(1/32,0.25),(1/16,2),(1/8,4),(0.25,0.25),(0.25,1),(0.25,4),(1,0.25),(1,4),(4,1),(4,4),(1,1)]
        rows=[dict(arm=arm,sample_rate=sr,channels=ch,boundary_stress=True,callbacks=10000,boundary_tempo_pitch=ratios)
              for arm,sr,ch in itertools.product(['rust','apple'],[44100,48000,96000],[1,2])]
        self.assertTrue(complete_boundary_protocol(rows,['rust','apple']))
        rows[-1]['boundary_tempo_pitch']=ratios[4:]
        self.assertFalse(complete_boundary_protocol(rows,['rust','apple']))
        rows[-1]['boundary_tempo_pitch']=ratios
        rows[-1]['boundary_stress']=False
        self.assertFalse(complete_boundary_protocol(rows,['rust','apple']))
        rows[-1].update(boundary_stress=True,callbacks=1000)
        self.assertFalse(complete_boundary_protocol(rows,['rust','apple']))

    def test_percentile_support_and_deadline(self):
        row = perf_row(
            dict(times_ns=[1, 2, 3, 40000000], callback=64, sample_rate=48000)
        )
        self.assertIsNone(row["p999_ns"])
        self.assertEqual(row["deadline_misses"], 1)
        self.assertEqual(
            perf_row(dict(times_ns=list(range(10000)), callback=64, sample_rate=48000))[
                "p999_ns"
            ],
            9989,
        )

    def test_deterministic_antiphase(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.wav", Path(d) / "b.wav"
            synthetic(a, 44100, "antiphase")
            synthetic(b, 44100, "antiphase")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            ch, sr, data = wav_read(a)
            self.assertEqual((ch, sr), (2, 44100))
            self.assertTrue(all(l == -r for l, r in zip(data[::2], data[1::2])))

    def test_trace_does_not_align_away_a_difference(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            row = dict(
                consumed=128,
                output=64,
                frame=1,
                marks=[[0]],
                spectrum_hash="1",
                magnitude_hash="2",
                classifier=[0],
            )
            a.write_text(json.dumps(row) + "\n")
            row["output"] = 65
            row["marks"] = [[1]]
            b.write_text(json.dumps(row) + "\n")
            self.assertEqual(first_difference(a, b)["stage"], "output")

    def test_mark_score_drift_is_separate_from_position_and_classifier_changes(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            reference=dict(consumed=128,output=64,frame=1,marks=[[42,100]],spectrum_hash='1',magnitude_hash='2',classifier=[0])
            score_only={**reference,'marks':[[42,101]]}
            position_change={**score_only,'marks':[[43,101]],'classifier':[1]}
            a.write_text(json.dumps(reference)+'\n'+json.dumps(reference)+'\n')
            b.write_text(json.dumps(score_only)+'\n'+json.dumps(position_change)+'\n')
            result=first_difference(a,b)
            self.assertEqual((result['stage'],result['step']),('marks',0))
            self.assertEqual(result['first_by_stage']['mark_positions']['step'],1)
            self.assertEqual(result['first_by_stage']['classifier']['step'],1)

    def test_trace_includes_synthesis_stages_and_output_samples(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            reference=dict(consumed=128,output=64,frame=1,marks=[],spectrum_hash='1',magnitude_hash='2',classifier=[0],stages={'output_phase':['3']},output_hash=['4'])
            candidate={**reference,'stages':{'output_phase':['5']},'output_hash':['6']}
            a.write_text(json.dumps(reference)+'\n')
            b.write_text(json.dumps(candidate)+'\n')
            result=first_difference(a,b)
            self.assertEqual(result['stage'],'stages.output_phase')
            self.assertEqual(result['first_by_stage']['output_hash']['step'],0)


if __name__ == "__main__":
    unittest.main()
