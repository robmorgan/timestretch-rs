`StreamSession` is the sole supported processing adapter. A session has one
owner and may move between threads; callers serialize its methods. It never
spawns a thread, sleeps, obtains audio from a callback or accesses a device.

1. Construct `SessionConfig`, then `PreparedSession::new`. Preparation validates
   rates/layout, backend, joint parameters and capacities, and allocates storage.
   `activate` transfers ownership without allocating. `config()` reports the
   resolved backend. `owned_memory()` reports native buffers and adapter
   storage; FFT-plan memory remains opaque and host queues are excluded.
2. `source_request()` returns an epoch and absolute source span. Supply any
   nonempty prefix, using interleaved finite f32 samples. Uneven and one-frame
   fragments are supported. A stale epoch/start or invalid geometry is rejected
   before input is consumed. Source positions may wrap at a continuous loop.
3. `advance(quota)` runs bounded stages until the quota or a dependency stops it.
   `NeedSource` requires input; `NeedOutput` requires collecting/acknowledging
   output; `NeedAcknowledgement` requires reading control events. `Priming`,
   `Progress`, `Paused`, `Rendered`, `Finished` and `Fault` are distinct.
   A stage is not a wall-clock deadline guarantee. Reference prefill and restart
   are still bounded indivisible operations and run on the desktop worker.
4. `read_output` returns an `OutputStamp` and writes only its reported frames.
   Unwritten host-buffer samples remain the host's responsibility. The session
   stores source mappings until consumption. It does not count collected output
   as playback. A host queues stamps alongside samples and returns cumulative
   `OutputConsumption` receipts after delivery. Receipts may be coalesced within
   an epoch, never across one. Repeated receipts are idempotent; older epochs
   cannot advance position; future/overrun receipts are errors.
5. `schedule(id, output_frame, parameters)` targets the session's output domain.
   `commit_horizon` separates already rendered audio. Commands apply at legal
   algorithm boundaries; `next_applied` reports requested/actual positions and
   achieved ratios. Past-horizon commands increment lateness telemetry. Both
   command and acknowledgement capacity are bounded; rejection is explicit.
6. Known source length establishes EOF automatically after its last frame.
   Otherwise `finish_input` is an ordered event at the current source cursor.
   Repeated finish is idempotent. The session supplies internal terminal zeros,
   keeps the reference's initial-unused trim, and retains achieved-ratio length.
   `rendered_finished` does not imply consumption finished. `finished` becomes
   true only after all retained audio is acknowledged. Dynamic parameters use
   piecewise achieved-duration accounting rather than a single nominal ratio.
7. Pause preserves DSP, queues and parameters. Seek requires `Seekable` source
   access, changes epoch, cancels superseded controls, invalidates queued media,
   computes history/origin, resets and primes while retaining pause state. Seek
   at EOF produces an empty remainder; successive seeks supersede earlier ones.
   Hosts invalidate their old queue with an O(1) boundary handoff. Old input and
   receipts cannot enter the new epoch. Do not perform restart on a device thread.
8. A loop region is half-open `[start,end)` and requires seekable input. Enabling
   or disabling a region re-enters at the consumed position through the seek
   transaction. Once running, wraps preserve DSP history and source mappings;
   they do not reset the processor. A seek beyond an active loop end is rejected;
   disable the region before seeking outside it.
9. Prepare backend replacements away from the callback. `replace_prepared`
   preserves formats, consumed position, loop, selected parameters and transport,
   advances the epoch and returns retired storage. Drop retired storage off the
   audio thread. The host supplies newly requested source and waits for its
   chosen reserve before resuming delivery. The library performs the media
   transaction; the desktop chooses worker wakeups and reserve.

Position is in source frames corresponding to delivered stream samples. It is
not a claim about physical speaker presentation; device latency/timestamps are
host information. Timing fields name their rate domains. Alignment delay is
zero after the reference startup trim; converter lookahead is availability
latency, not an extra sample shift. `control_quantum_frames` is a conservative
maximum output block between control boundaries, not a promise of exact
sample control. Add both session pending output and the host queue reserve when
reporting control latency.

The public limits are mono/stereo; source/output 32–192 kHz; tempo 1/32–4,
duration 0.25–32 and pitch 0.25–4 with 0.25 <= pitch/tempo <= 32; 65,536–1,048,576 maximum inflight
frames and 1–4,096 controls. Inflight accounting includes pending and collected
but unacknowledged output. Sequential live sources support normal processing
and EOF; random access is never inferred from a borrowed slice.
