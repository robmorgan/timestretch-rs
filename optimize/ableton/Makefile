.PHONY: ableton-refs library-refs score-ableton compare full manifest loop loop-dry loop-resume clean help

SCRIPT_DIR := scripts
CATALOG := track_catalog.json
ABLETON_REF_DIR := refs/ableton
LIBRARY_REF_DIR := refs/library
LOG_DIR := logs
MANIFEST := ableton_manifest.json
PYTHON := ../.venv/bin/python

ableton-refs:    ## Generate Ableton Complex Pro references (requires Ableton + macOS)
	$(PYTHON) $(SCRIPT_DIR)/batch_generate_refs.py $(CATALOG) $(ABLETON_REF_DIR) \
		--skip-existing --manifest-out $(MANIFEST)

library-refs:    ## Generate library outputs for same tracks/ratios
	$(PYTHON) $(SCRIPT_DIR)/generate_library_refs.py $(CATALOG) $(LIBRARY_REF_DIR) \
		--skip-existing --manifest-out library_manifest.json

score-ableton:   ## Score library outputs vs Ableton references
	cd .. && .venv/bin/python scripts/score.py --ref-source ableton \
		--ableton-manifest ableton/$(MANIFEST) \
		--batch ableton/$(LOG_DIR)/scores.json

compare:         ## Run library refs + score (no Ableton rendering)
	$(MAKE) library-refs
	$(MAKE) score-ableton

full:            ## Full pipeline: Ableton refs + library refs + score
	$(MAKE) ableton-refs
	$(MAKE) compare

manifest:        ## Generate Ableton manifest without rendering
	$(PYTHON) $(SCRIPT_DIR)/batch_generate_refs.py $(CATALOG) $(ABLETON_REF_DIR) \
		--dry-run --manifest-out $(MANIFEST)

loop:            ## Run Ableton optimization loop
	./scripts/optimize_loop.sh

loop-dry:        ## Score only (no agent)
	./scripts/optimize_loop.sh --dry-run

loop-resume:     ## Resume interrupted Ableton optimization
	./scripts/optimize_loop.sh --resume

clean:           ## Clean generated outputs and logs
	rm -rf $(LIBRARY_REF_DIR)/*.wav $(LOG_DIR)/* $(MANIFEST) library_manifest.json

help:            ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-17s\033[0m %s\n", $$1, $$2}'
