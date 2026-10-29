//! Preallocated native-core timing with generated inputs; startup and source
//! accounting are identical across numerical providers.
use super::*;

pub(super) fn fixture(path: &str, ch: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() || bytes.len() % (4 * ch) != 0 {
        return Err("fixture must contain whole interleaved f32le frames".into());
    }
    let samples: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if !samples.iter().all(|x| x.is_finite()) {
        return Err("non-finite fixture".into());
    }
    Ok(samples)
}
fn hash(h: &mut u64, value: u64) {
    *h = (*h ^ value).wrapping_mul(1099511628211);
}
pub(super) fn perf(a: &[String]) -> Result<Value> {
    let arm = &a[0];
    let sr: u32 = a[1].parse()?;
    let ch: usize = a[2].parse()?;
    let cb: usize = a[3].parse()?;
    let tempo: f32 = a[4].parse()?;
    let n: usize = a[5].parse()?;
    let audit = a.get(7).is_some_and(|x| x == "audit");
    if n == 0 || cb == 0 || ![1, 2].contains(&ch) || !tempo.is_finite() || tempo <= 0. || tempo > 2.
    {
        return Err("invalid core workload".into());
    }
    allocator::init();
    let before = allocator::live();
    let start = Instant::now();
    let mut p = NativeProcessor::new_with_numerics(sr, ch, numerics(arm)?)?;
    let init_ns = start.elapsed().as_nanos() as u64;
    let owned = allocator::live() - before;
    let (status, actual, _) = p.set_ratio(1. / tempo, 1., RatioAdjustment::Stretch, true)?;
    if status != 0 {
        return Err("ratio rejected".into());
    }
    let fixture = fixture(&a[6], ch)?;
    let source_size = (n + 64) * cb * 3 + p.block_size() * 16;
    let source: [Vec<f32>; 2] = std::array::from_fn(|c| {
        (0..source_size)
            .map(|i| fixture[(i % (fixture.len() / ch)) * ch + c.min(ch - 1)])
            .collect()
    });
    let mut out = [vec![0.; p.output_capacity()], vec![0.; p.output_capacity()]];
    let pre = p.prefill_frames();
    let iv = [&source[0][..pre], &source[1][..pre]];
    let [l, r] = &mut out;
    let mut ov = [l.as_mut_slice(), r.as_mut_slice()];
    let pre_output = p.prefill(&iv[..ch], Some(&mut ov[..ch]))?;
    let mut cursor = pre;
    let mut pending = 0;
    let mut times = vec![0u64; n];
    let mut calls = 0;
    let mut schedule_hash = 14695981039346656037u64;
    let mut audio_hash = schedule_hash;
    let mut allocations = 0;
    let mut system_allocations = allocator::system_count().map(|_| 0);
    for i in 0..n + 64 {
        allocator::begin();
        let start = Instant::now();
        while pending < cb {
            let needed = p.frames_needed(0)?;
            if cursor + needed > source_size {
                return Err("source exhausted".into());
            }
            let iv = [
                &source[0][cursor..cursor + needed],
                &source[1][cursor..cursor + needed],
            ];
            let [l, r] = &mut out;
            let mut ov = [l.as_mut_slice(), r.as_mut_slice()];
            let count = p.process(&iv[..ch], &mut ov[..ch])?;
            if count == 0 {
                return Err("core made no progress".into());
            }
            cursor += needed;
            pending += count;
            calls += 1;
            hash(&mut schedule_hash, needed as u64);
            hash(&mut schedule_hash, count as u64);
            if audit {
                for plane in &out[..ch] {
                    for x in &plane[..count] {
                        hash(&mut audio_hash, u64::from(x.to_bits()));
                    }
                }
            }
        }
        pending -= cb;
        let ns = start.elapsed().as_nanos() as u64;
        let alloc = allocator::end();
        if i >= 64 {
            times[i - 64] = ns;
            allocations += alloc;
            if let (Some(total), Some(count)) = (&mut system_allocations, allocator::system_count())
            {
                *total += count;
            }
        }
        std::hint::black_box(&out);
    }
    Ok(
        json!({"arm":format!("{arm}_core"),"sample_rate":sr,"channels":ch,"callback":cb,"tempo":tempo,"callbacks":n,"warmup":64,"actual_stretch":actual,"init_ns":init_ns,"owned_rust_bytes":owned,"allocations":allocations,"system_allocations":system_allocations,"calls":calls,"consumed":cursor,"pending":pending,"prefill_input":pre,"prefill_output":pre_output,"schedule_hash":schedule_hash.to_string(),"audio_hash":audit.then(||audio_hash.to_string()),"audit":audit,"times_ns":times,"scope":"preallocated Direct core; excludes public wrapper, graph and device"}),
    )
}
