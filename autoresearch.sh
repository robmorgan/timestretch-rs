#!/bin/bash
set -euo pipefail

# Quick compile check
cargo build --features qa-harnesses --release --test streaming_quality 2>&1 | tail -3

# Run the streaming quality benchmark 3 times, take median
declare -a scores
declare -a perc_scores
declare -a edm_scores
declare -a harm_scores

for run in 1 2 3; do
    output=$(cargo test --features qa-harnesses --release --test streaming_quality -- --nocapture 2>&1)

    score=$(echo "$output" | grep "^METRIC quality_score=" | sed 's/METRIC quality_score=//')

    # Parse per-signal-type composites
    perc_avg=$(echo "$output" | grep "^percussive" | awk '{print $NF}' | awk '{s+=$1; c++} END {printf "%.1f", s/c*1000}')
    edm_avg=$(echo "$output" | grep "^edm" | awk '{print $NF}' | awk '{s+=$1; c++} END {printf "%.1f", s/c*1000}')
    harm_avg=$(echo "$output" | grep "^harmonic" | awk '{print $NF}' | awk '{s+=$1; c++} END {printf "%.1f", s/c*1000}')

    scores+=("$score")
    perc_scores+=("$perc_avg")
    edm_scores+=("$edm_avg")
    harm_scores+=("$harm_avg")
done

# Sort and take median (2nd of 3)
sorted_scores=($(printf '%s\n' "${scores[@]}" | sort -n))
sorted_perc=($(printf '%s\n' "${perc_scores[@]}" | sort -n))
sorted_edm=($(printf '%s\n' "${edm_scores[@]}" | sort -n))
sorted_harm=($(printf '%s\n' "${harm_scores[@]}" | sort -n))

median_score=${sorted_scores[1]}
median_perc=${sorted_perc[1]}
median_edm=${sorted_edm[1]}
median_harm=${sorted_harm[1]}

echo "All runs: ${scores[*]}"
echo "METRIC quality_score=${median_score}"
echo "METRIC percussive_score=${median_perc}"
echo "METRIC edm_score=${median_edm}"
echo "METRIC harmonic_score=${median_harm}"
