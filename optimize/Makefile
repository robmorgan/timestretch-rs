.PHONY: refs score loop report clean setup install-deps install-tools resume

REPO_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

refs:            ## Generate reference files
	./scripts/generate_references.sh

score:           ## Run scoring only (no optimization)
	./scripts/optimize_loop.sh --dry-run

loop:            ## Run full optimization loop
	./scripts/optimize_loop.sh

resume:          ## Resume interrupted optimization
	./scripts/optimize_loop.sh --resume

report:          ## Generate optimization report
	python3 scripts/report.py

clean:           ## Clean outputs and logs
	rm -rf outputs/* logs/* references/*

setup:           ## Create venv and install all dependencies
	uv venv .venv
	$(MAKE) install-deps
	$(MAKE) install-tools

install-deps:    ## Install Python dependencies
	uv pip install librosa soundfile numpy matplotlib pandas tabulate

install-tools:   ## Install reference audio tools
	@OS=$$(uname -s); \
	if [ "$$OS" = "Darwin" ]; then \
		echo "Installing via Homebrew..."; \
		brew install rubberband soundtouch sox; \
	elif [ -f /etc/arch-release ]; then \
		echo "Installing via pacman..."; \
		sudo pacman -S --needed rubberband soundtouch sox; \
	elif [ -f /etc/debian_version ]; then \
		echo "Installing via apt..."; \
		sudo apt install -y rubberband-cli soundstretch sox; \
	else \
		echo "Unsupported OS. Install manually: rubberband, soundtouch, sox"; \
		exit 1; \
	fi

help:            ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
