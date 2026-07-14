#!/usr/bin/env bash
# Fetches the public, redistributable DJ corpus (ROADMAP Stage 7) into
# benchmarks/audio/public-corpus/ and verifies every file against its
# pinned SHA-256. Idempotent: files that already verify are skipped.
#
# All tracks are Creative Commons netlabel releases hosted on the
# Internet Archive (stable, unauthenticated URLs). Licenses per track are
# recorded in benchmarks/manifest.toml next to the corpus entries; every
# license permits redistribution and derivatives (CC0 / CC-BY / CC-BY-SA),
# so stretched renders and CI artifacts are safe to publish with
# attribution.
#
# Ground-truth tempos were verified with two independent estimators
# (the repo detector and a coherent onset-envelope Goertzel scan over the
# full track; all scored entries peak at exact integer BPM with agreement
# across four 60 s windows). See the manifest comments.

set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/benchmarks/audio/public-corpus"
mkdir -p "$DIR"

fetch() {
    local sha="$1" item="$2" file="$3"
    local dest="$DIR/$file"
    if [[ -f "$dest" ]] && echo "$sha  $dest" | shasum -a 256 -c - >/dev/null 2>&1; then
        echo "ok       $file"
        return
    fi
    echo "fetching $file"
    curl -fsSL --retry 3 -o "$dest" "https://archive.org/download/$item/$file"
    echo "$sha  $dest" | shasum -a 256 -c - >/dev/null || {
        echo "CHECKSUM MISMATCH: $file" >&2
        rm -f "$dest"
        exit 1
    }
}

# Scored BPM-corpus entries (verified integer tempos).
fetch 412721bc9404255639c1f929f973b9cb28dd3243dacd68b83613f6c1b98b9ee3 yarn020 "01-Valya_Kan-December.mp3"
fetch 385e2f65d59019a7185f9668f63ae47b32440140428633f1b5a5f2a8882f97d7 yarn020 "02-Valya_Kan-Memories.mp3"
fetch 2c5ac35b903c7bd27870d4544e2f84571f7d1f13064fa75703e298908412ae1f yarn021 "01-Mehta-Stay_Like_That.mp3"
fetch e9ffaf3e93107b336023bcdbc8e0be2328378791a7728b041e0ef0b747aee643 SICMON010 "33RD_RATE_REVS_-_X_sicmon010.mp3"
fetch 50aa9a376fe2e84c050da5a7f2c642792e04b91a30b9d012f9e14ac136ebf0bc SICMON012 "33RD_RATE_REVS_-_Group_Therapy_sicmon012.mp3"
fetch dd947b758985479d19728f2fb345ec2a9b5f2c9fb9f4251290c7744cd306f984 mtrnc002 "mtrnc002_01_-_J-Lab_-_Radiophonique.mp3"
fetch 61de8728f4c780be6ddcd47784cb70894b533795b5a919425abf6d85c0ae52ae unfound53 "unfound53_01_-_raymundo_mendoza_-_wrist_watch.mp3"

# Rhythmically ambiguous entries (swung UK garage, broken techno):
# tempos owner-ear-verified 2026-07-14 (126 / 125), scored like the rest.
fetch cb3a303bd35bc92c27ddfec44db8666c0323913e67e2f8050282f6349651b23a yarn015 "01-Interplanetary_Criminal-Saucers.mp3"
fetch 65a82156174ec368355da9b95a118878bf200b1128acdbf725674ffa76e1952e yarn022 "01-Raptom_Snig-Arf.mp3"

echo "public corpus complete: $DIR"
