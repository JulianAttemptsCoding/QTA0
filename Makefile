# TACTIC-MoB v4 Makefile (sec2.3, sec18). On Windows without GNU make, use: python make.py <target>
PY ?= python
CLI = $(PY) -m tactic.cli

.PHONY: setup test data-alpaca data-factors data-universe day1-audits universe costs \
        labels features diagnostics baselines infra train aggregate decide backtest \
        stress firewall reports all lint

setup:        ; $(CLI) setup
test:         ; $(PY) -m pytest -q
data-alpaca:  ; $(CLI) data-alpaca
data-factors: ; $(CLI) data-factors
data-universe:; $(CLI) data-universe
day1-audits:  ; $(CLI) day1-audits
universe:     ; $(CLI) universe
costs:        ; $(CLI) costs
labels:       ; $(CLI) labels
features:     ; $(CLI) features
diagnostics:  ; $(CLI) diagnostics
baselines:    ; $(CLI) baselines
infra:        ; $(CLI) infra
train:        ; $(CLI) train
aggregate:    ; $(CLI) aggregate
decide:       ; $(CLI) decide
backtest:     ; $(CLI) backtest
stress:       ; $(CLI) stress
firewall:     ; $(CLI) firewall
reports:      ; $(CLI) reports
all:          ; $(CLI) all
lint:         ; $(PY) -m ruff check src tests
