SHELL := /bin/bash

PYTHON := .venv/bin/python
RUN ?= full_001
CHECKPOINT ?= checkpoints/$(RUN)/final.pt
UPDATES ?= 500
ENVIRONMENTS ?= 512
ROLLOUT_STEPS ?= 256
GAE_LAMBDA ?= 0.99
UPDATE_EPOCHS ?= 4
MINIBATCH_STATES ?= 2048
LEARNING_RATE ?= 0.0002
MINIMUM_LEARNING_RATE ?= 0.00001
ROLLBACK_LEARNING_RATE_FACTOR ?= 0.5
MAX_CONSECUTIVE_KL_ROLLBACKS ?= 8
TARGET_KL ?= 0.02
ROLLBACK_KL ?= 0.05
CURRICULUM_UPDATES ?= 100
CURRICULUM_START_SPEED ?= 0.4
CURRICULUM_SPEED_STEP ?= 0.01
CURRICULUM_LATE_SPEED_STEP ?= 0.01
CURRICULUM_LATE_SPEED_THRESHOLD ?= 0.30
CURRICULUM_SAVED_THRESHOLD ?= 9.8
CURRICULUM_FAST_THRESHOLD ?= 0.90
CURRICULUM_PERFECT_THRESHOLD ?= 0.90
CURRICULUM_WINDOW ?= 20
EPISODES ?= 4096
SEED ?= 10000
REPLAY_SEED ?= 123
PLAYBACK_RATE ?= 0.5
PLOT_EVERY ?= 5
SAVE_EVERY ?= 25
VALIDATE_EVERY ?= 10
VALIDATION_EPISODES ?= 1024
VALIDATION_ROLLBACK_DROP ?= 0.75
VALIDATION_PERFECT_ROLLBACK_DROP ?= 0.20
DEVICE ?= cuda
INITIALIZE_ACTOR_FROM ?=
INITIALIZE_ARG = $(if $(strip $(INITIALIZE_ACTOR_FROM)),--initialize-actor-from $(INITIALIZE_ACTOR_FROM),)
RESUME ?=
RESUME_ARG = $(if $(strip $(RESUME)),--resume $(RESUME),)

.DEFAULT_GOAL := help

.PHONY: help check setup gpu smoke train train-fast train-curriculum eval eval-smoke holdout baselines replay replay-smoke progress

help:
	@printf '%s\n' \
	  'Sharks & Minnows commands' \
	  '' \
	  '  make smoke              Train a 50-update smoke model' \
	  '  make eval-smoke         Evaluate checkpoints/smoke/final.pt' \
	  '  make replay-smoke       Watch the smoke model play' \
	  '  make train RUN=run_001  Train a full model' \
	  '  make train-fast RUN=run_001  ~40% faster updates (2 PPO passes)' \
	  '  make train-curriculum RUN=run_001  Performance-gated curriculum with PPO safety' \
	  '  make eval RUN=run_001   Evaluate a trained model' \
	  '  make replay RUN=run_001 Watch a trained model' \
	  '  make holdout RUN=run_001 10,000-game final test' \
	  '  make baselines          Evaluate fixed baselines only' \
	  '  make progress RUN=run_001 Open the training chart' \
	  '  make gpu                Verify CUDA and GPU identity' \
	  '' \
	  'Override defaults, for example:' \
	  '  make train RUN=experiment_2 UPDATES=1000 ENVIRONMENTS=1024' \
	  '  make replay RUN=experiment_2 REPLAY_SEED=456 PLAYBACK_RATE=0.25'
	@printf '%s\n' \
	  '  make train-curriculum RUN=run_001 UPDATES=2000 RESUME=checkpoints/run_001/latest.pt'

check:
	@test -x $(PYTHON) || { echo 'Missing .venv. Run: make setup'; exit 1; }

setup:
	UV_CACHE_DIR=/tmp/shark-uv-cache uv sync

gpu: check
	$(PYTHON) -c 'import torch; print("cuda:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'

smoke: check
	$(PYTHON) train.py --updates 50 --environments 256 --rollout-steps $(ROLLOUT_STEPS) --gae-lambda $(GAE_LAMBDA) --update-epochs $(UPDATE_EPOCHS) --minibatch-states $(MINIBATCH_STATES) --device $(DEVICE) --plot-every 5 --save-every 25 --checkpoint-dir checkpoints/smoke

train: check
	$(PYTHON) train.py --updates $(UPDATES) --environments $(ENVIRONMENTS) --rollout-steps $(ROLLOUT_STEPS) --gae-lambda $(GAE_LAMBDA) --update-epochs $(UPDATE_EPOCHS) --minibatch-states $(MINIBATCH_STATES) --learning-rate $(LEARNING_RATE) --minimum-learning-rate $(MINIMUM_LEARNING_RATE) --rollback-learning-rate-factor $(ROLLBACK_LEARNING_RATE_FACTOR) --max-consecutive-kl-rollbacks $(MAX_CONSECUTIVE_KL_ROLLBACKS) --target-kl $(TARGET_KL) --rollback-kl $(ROLLBACK_KL) $(RESUME_ARG) $(INITIALIZE_ARG) --device $(DEVICE) --plot-every $(PLOT_EVERY) --save-every $(SAVE_EVERY) --validate-every $(VALIDATE_EVERY) --validation-episodes $(VALIDATION_EPISODES) --checkpoint-dir checkpoints/$(RUN)

train-fast: check
	$(PYTHON) train.py --updates $(UPDATES) --environments $(ENVIRONMENTS) --rollout-steps $(ROLLOUT_STEPS) --gae-lambda $(GAE_LAMBDA) --update-epochs 2 --minibatch-states $(MINIBATCH_STATES) --learning-rate $(LEARNING_RATE) --minimum-learning-rate $(MINIMUM_LEARNING_RATE) --rollback-learning-rate-factor $(ROLLBACK_LEARNING_RATE_FACTOR) --max-consecutive-kl-rollbacks $(MAX_CONSECUTIVE_KL_ROLLBACKS) --target-kl $(TARGET_KL) --rollback-kl $(ROLLBACK_KL) $(RESUME_ARG) $(INITIALIZE_ARG) --device $(DEVICE) --plot-every 25 --save-every 50 --validate-every $(VALIDATE_EVERY) --validation-episodes $(VALIDATION_EPISODES) --checkpoint-dir checkpoints/$(RUN)

train-curriculum: check
	$(PYTHON) train.py --updates $(UPDATES) --environments $(ENVIRONMENTS) --rollout-steps $(ROLLOUT_STEPS) --gae-lambda $(GAE_LAMBDA) --update-epochs $(UPDATE_EPOCHS) --minibatch-states $(MINIBATCH_STATES) --learning-rate $(LEARNING_RATE) --minimum-learning-rate $(MINIMUM_LEARNING_RATE) --rollback-learning-rate-factor $(ROLLBACK_LEARNING_RATE_FACTOR) --max-consecutive-kl-rollbacks $(MAX_CONSECUTIVE_KL_ROLLBACKS) --target-kl $(TARGET_KL) --rollback-kl $(ROLLBACK_KL) --adaptive-curriculum --curriculum-start-slow-speed $(CURRICULUM_START_SPEED) --curriculum-speed-step $(CURRICULUM_SPEED_STEP) --curriculum-late-speed-step $(CURRICULUM_LATE_SPEED_STEP) --curriculum-late-speed-threshold $(CURRICULUM_LATE_SPEED_THRESHOLD) --curriculum-saved-threshold $(CURRICULUM_SAVED_THRESHOLD) --curriculum-fast-survival-threshold $(CURRICULUM_FAST_THRESHOLD) --curriculum-perfect-threshold $(CURRICULUM_PERFECT_THRESHOLD) --curriculum-window $(CURRICULUM_WINDOW) --validation-rollback-drop $(VALIDATION_ROLLBACK_DROP) --validation-perfect-rollback-drop $(VALIDATION_PERFECT_ROLLBACK_DROP) $(RESUME_ARG) $(INITIALIZE_ARG) --device $(DEVICE) --plot-every $(PLOT_EVERY) --save-every $(SAVE_EVERY) --validate-every $(VALIDATE_EVERY) --validation-episodes $(VALIDATION_EPISODES) --checkpoint-dir checkpoints/$(RUN)

eval: check
	$(PYTHON) evaluate.py --checkpoint $(CHECKPOINT) --episodes $(EPISODES) --device $(DEVICE) --seed $(SEED) --output-dir evaluation/$(RUN)

eval-smoke: check
	$(MAKE) eval RUN=smoke CHECKPOINT=checkpoints/smoke/final.pt

holdout: check
	$(PYTHON) evaluate.py --checkpoint $(CHECKPOINT) --episodes 10000 --device $(DEVICE) --seed 20000 --output-dir evaluation/$(RUN)_holdout

baselines: check
	$(PYTHON) evaluate.py --episodes $(EPISODES) --device $(DEVICE) --seed $(SEED) --output-dir evaluation/baselines

replay: check
	$(PYTHON) replay.py --checkpoint $(CHECKPOINT) --seed $(REPLAY_SEED) --playback-rate $(PLAYBACK_RATE)

replay-smoke: check
	$(MAKE) replay RUN=smoke CHECKPOINT=checkpoints/smoke/final.pt

progress: check
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open checkpoints/$(RUN)/training_progress.png; \
	else \
		echo "Open checkpoints/$(RUN)/training_progress.png"; \
	fi
