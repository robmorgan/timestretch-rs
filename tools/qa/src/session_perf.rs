//! Unpaced processing cost of the public adapter; no queue/device/wakeup timer.
use super::*;
use timestretch::{PreparedSession, ProcessStatus, SourceAccess, TempoRate};
pub(super) fn run(a: &[String]) -> Result<Value> {
    let arm = &a[0];
    let sr: u32 = a[1].parse()?;
    let ch: usize = a[2].parse()?;
    let cb: usize = a[3].parse()?;
    let tempo: f64 = a[4].parse()?;
    let n: usize = a[5].parse()?;
    let stress = a.get(6).is_some_and(|x| x == "stress");
    if n == 0 || cb == 0 || cb > 8192 || ![1, 2].contains(&ch) {
        return Err("invalid session workload".into());
    }
    let source: Vec<_> = (0..sr as usize)
        .flat_map(|i| (0..ch).map(move |c| (i as f32 * (0.031 + c as f32 * 0.013)).sin() * 0.2))
        .collect();
    allocator::init();
    let t = Instant::now();
    let prepared = PreparedSession::new(SessionConfig {
        backend: backend(arm)?,
        channels: ch,
        source_sample_rate: sr,
        output_sample_rate: sr,
        source_frames: Some(sr as u64),
        source_access: SourceAccess::Seekable,
        parameters: Parameters::keylock(TempoRate::new(tempo)?),
        ..Default::default()
    })?;
    let init_ns = t.elapsed().as_nanos() as u64;
    // Provider ownership must survive a move to the host's execution thread.
    std::thread::spawn(move||->std::result::Result<Value,String>{
        let mut s=prepared.activate();s.set_loop(Some((0,sr as u64))).map_err(|e|e.to_string())?;
        let mut output=vec![0.0;cb*ch];let mut times=vec![0u64;n];let mut allocations=0;let mut frees=0;let mut system=allocator::system_count().map(|_|0usize);
        let mut startup_ns=0;let mut largest_stage_ns=0;let mut finite=true;
        for i in 0..n+64 {
            allocator::begin();let t=Instant::now();
            if stress && i%1000==0 {
                s.seek((i as u64*997)%sr as u64).map_err(|e|e.to_string())?;
                let rate=if i%2000==0 {0.5}else{2.0};
                s.schedule(i as u64,s.commit_horizon(),Parameters::linked(TempoRate::new(rate).unwrap()).unwrap()).map_err(|e|e.to_string())?;
            }
            let mut filled=0;
            while filled<cb {
                let stage=Instant::now();let report=s.advance(1);largest_stage_ns=largest_stage_ns.max(stage.elapsed().as_nanos()as u64);
                if report.status==ProcessStatus::Fault {return Err(format!("{:?}",s.fault()));}
                if report.status==ProcessStatus::NeedSource {
                    let r=s.source_request().unwrap();s.supply(r,&source[r.start_frame as usize*ch..(r.start_frame as usize+r.frames)*ch]).map_err(|e|e.to_string())?;
                }
                let stamp=s.read_output(&mut output[filled*ch..]).map_err(|e|e.to_string())?;
                filled+=stamp.frames;s.acknowledge(stamp.consumed(stamp.frames).unwrap()).map_err(|e|e.to_string())?;
                while s.next_applied().is_some(){}
            }
            let ns=t.elapsed().as_nanos()as u64;let count=allocator::end();let dealloc=allocator::deallocations();
            allocations+=count;frees+=dealloc;
            if let(Some(total),Some(count))=(&mut system,allocator::system_count()){*total+=count;}
            if i==0{startup_ns=ns;} else if i>=64{times[i-64]=ns;}
            finite &= output.iter().all(|x|x.is_finite());
        }
        let memory=s.owned_memory();
        Ok(json!({"sample_rate":sr,"channels":ch,"callback":cb,"tempo":tempo,"callbacks":n,"warmup":64,"stress":stress,
            "init_ns":init_ns,"startup_ns":startup_ns,"largest_indivisible_stage_ns":largest_stage_ns,"allocations":allocations,"deallocations":frees,"system_allocations":system,
            "owned_rust_bytes":memory.adapter_bytes+memory.native_buffer_bytes,"opaque_fft_plan_bytes":memory.fft_plan_bytes,"finite":finite,"times_ns":times,
            "scope":"public StreamSession moved to host thread; unpaced processing, source supply and consumption receipts; allocation audit includes startup/seek; excludes device and worker wakeup"}))
    }).join().map_err(|_|"session worker panicked")?.map_err(Into::into)
}
