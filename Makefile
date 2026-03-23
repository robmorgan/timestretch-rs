MODEL ?= gpt-5.4
ITERATIONS ?= 50
STATE_DIR ?= .codex-loop
TEST_CMD ?= cargo test --test realtime_allocations --test realtime_dj_conditions
PLAN_MODEL ?= $(MODEL)
PLAN_ITERATIONS ?= $(ITERATIONS)
PLAN_STATE_DIR ?= .codex-plan-loop
PLAN_SMOKE_CMD ?= cargo test --test realtime_allocations --test realtime_dj_conditions

.PHONY: loop resume plan-loop plan-resume plan-status test-fast clean-loop clean-plan-loop help

loop: ## Run a bounded roadmap implementation loop
	@MODEL="$(MODEL)" ITERATIONS="$(ITERATIONS)" STATE_DIR="$(STATE_DIR)" TEST_CMD="$(TEST_CMD)" ./scripts/roadmap_loop.sh start

resume: ## Resume the saved roadmap loop session
	@MODEL="$(MODEL)" ITERATIONS="$(ITERATIONS)" STATE_DIR="$(STATE_DIR)" TEST_CMD="$(TEST_CMD)" ./scripts/roadmap_loop.sh resume

plan-loop: ## Run the active ROADMAP stage implementation loop
	@MODEL="$(PLAN_MODEL)" ITERATIONS="$(PLAN_ITERATIONS)" STATE_DIR="$(PLAN_STATE_DIR)" ./scripts/roadmap_loop.sh start --prompt-scope active-stage --failure-policy continue --stall-limit 5 --status-headings --stage-tests auto --smoke-cmd "$(PLAN_SMOKE_CMD)"

plan-resume: ## Resume the active ROADMAP stage implementation loop
	@MODEL="$(PLAN_MODEL)" ITERATIONS="$(PLAN_ITERATIONS)" STATE_DIR="$(PLAN_STATE_DIR)" ./scripts/roadmap_loop.sh resume --prompt-scope active-stage --failure-policy continue --stall-limit 5 --status-headings --stage-tests auto --smoke-cmd "$(PLAN_SMOKE_CMD)"

plan-status: ## Show ROADMAP plan-loop status
	@STATE_DIR="$(PLAN_STATE_DIR)" ./scripts/roadmap_loop.sh status --status-headings

test-fast: ## Run the fast continuous-improvement test suite
	@$(TEST_CMD)

clean-loop: ## Remove roadmap loop state and logs
	@rm -rf "$(STATE_DIR)"

clean-plan-loop: ## Remove plan-loop state and logs
	@rm -rf "$(PLAN_STATE_DIR)"

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'
