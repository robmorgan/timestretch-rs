# Recorded results

`historical-core.json` records the scope and binary hashes of the earlier local
DSP/desktop campaign. Historical results do not qualify a newly built executable.
Public-corpus BPM results and annotations are separate optional integration data.

New qualification writes raw JSON, CSV, source/binary hashes and HTML to a fresh
output directory. `tools/archive_qualification.py` can preserve a completed local
run outside disposable build output. Neither these older measurements nor an
external implementation is required to build or test the current library.
