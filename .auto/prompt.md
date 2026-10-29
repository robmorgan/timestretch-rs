# timestretch quality work

The production processor is timestretch. Use the single `StreamSession` API;
there is no profile selector, old engine graph or library-owned worker thread.
The accepted architecture is in docs/CORE_ARCHITECTURE.md and contracts
are in docs/SESSION_CONTRACT.md.

`./.auto/checks.sh` runs workspace Clippy, formatting and library checks.
`./.auto/measure.sh` emits the retained quality-gate JSON and fails any bound.
Use tools/qualify.py for fixture, quality and performance,
allocation, device and listening evidence. Keep evaluator and numerical bounds
fixed. Do not special-case fixtures, normalize scored output, align errors away
or loosen deadlines. Separate algorithm changes from backend substitutions.
