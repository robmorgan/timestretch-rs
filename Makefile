MODEL ?= gpt-5.4
ITERATIONS ?= 50
STATE_DIR ?= .codex-loop
TEST_CMD ?= cargo test --test dual_plane_rt --test realtime_allocations --test realtime_dj_conditions

.PHONY: loop resume test-fast clean-loop help

loop: ## Run a bounded roadmap implementation loop
	@MODEL="$(MODEL)" ITERATIONS="$(ITERATIONS)" STATE_DIR="$(STATE_DIR)" TEST_CMD="$(TEST_CMD)" ./scripts/roadmap_loop.sh start

resume: ## Resume the saved roadmap loop session
	@MODEL="$(MODEL)" ITERATIONS="$(ITERATIONS)" STATE_DIR="$(STATE_DIR)" TEST_CMD="$(TEST_CMD)" ./scripts/roadmap_loop.sh resume

test-fast: ## Run the fast continuous-improvement test suite
	@$(TEST_CMD)

clean-loop: ## Remove roadmap loop state and logs
	@rm -rf "$(STATE_DIR)"

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'
