# ============================================================
#  VARIABLES
# ============================================================

VENV       := /tmp/.venv

PYTHON     := /bin/python3
PIP        := $(VENV)/bin/pip
UV := /tmp/.venv/bin/uv
DEF_ENV	:= UV_PROJECT_ENVIRONMENT=/tmp/.venv
MAIN 	   :=
CONFIG	   :=

# ------------------------------------------------------------
#  Ansi colors
# ------------------------------------------------------------

RESET		:=	\033[0m
GRAY		:=	\033[1;90m
RED		    := 	\033[1;91m
GREEN		:=	\033[1;92m
YELLOW		:=	\033[1;93m
BLUE		:=	\033[1;94m
MAGENTA		:=	\033[1;95m
CYAN		:=	\033[1;96m
WHITE		:=	\033[1;97m

# ------------------------------------------------------------
#  Additional commands
# ------------------------------------------------------------

ECHO		:=	@echo
FIND		:=	@find
RM		    :=	@/bin/rm -rf

# ============================================================
#  RULES
# ============================================================

.PHONY: install run debug clean lint lint-strict help checker

# ------------------------------------------------------------
#  Default target
# ------------------------------------------------------------

help:
	$(ECHO) ""
	$(ECHO) " $(WHITE)Available targets:$(RESET)"
	$(ECHO) ""
	$(ECHO) "     $(WHITE)install$(RESET)      Install project dependencies"
	$(ECHO) "     $(WHITE)run$(RESET)          Run the main script"
	$(ECHO) "     $(WHITE)debug$(RESET)        Run the main script with pdb"
	$(ECHO) "     $(WHITE)clean$(RESET)        Remove temporary files and caches"
	$(ECHO) "     $(WHITE)lint$(RESET)         Run flake8 + mypy (standard flags)"
	$(ECHO) "     $(WHITE)lint-strict$(RESET)  Run flake8 + mypy --strict"
	$(ECHO) "     $(WHITE)checker$(RESET)  	  Run the output checker"
	$(ECHO) ""

# ------------------------------------------------------------
#  install — create a virtual environment and install deps
# ------------------------------------------------------------

install:
	$(ECHO) ">>> Creating virtual environment …"
	python3 -m venv $(VENV)
	$(ECHO) ">>> Installing uv inside venv …"
	$(PIP) install --upgrade pip
	$(PIP) install uv
	$(ECHO) ">>> Syncing dependencies with uv …"
	UV_CACHE_DIR=/tmp/.uv-cache $(DEF_ENV) $(UV) sync
	$(ECHO) ">>> Done."

# ------------------------------------------------------------
#  run — execute the main script
# ------------------------------------------------------------

run:
	$(ECHO) ">>> Running $(MAIN) $(CONFIG)"
	$(PYTHON) $(MAIN) $(CONFIG)

# ------------------------------------------------------------
#  debug — launch the main script under pdb
# ------------------------------------------------------------

debug:
	$(ECHO) ">>> Launching $(MAIN) under pdb …"
	$(PYTHON) -m pdb $(MAIN) $(CONFIG)

# ------------------------------------------------------------
#  clean — remove byte-compiled files and tool caches
# ------------------------------------------------------------

clean:
	$(ECHO) "$(YELLOW)>>> Cleaning __pycache__$(RESET)"
	$(FIND) . -type d -name "__pycache__"   -exec rm -rf {} + 2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning *.pyc$(RESET)"
	$(FIND) . -type f -name "*.pyc"         -delete              2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning *.pyo$(RESET)"
	$(FIND) . -type f -name "*.pyo"         -delete              2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning .mypy_cache$(RESET)"
	$(FIND) . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning .ruff_cache$(RESET)"
	$(FIND) . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning .pytest_cache$(RESET)"
	$(FIND) . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	$(ECHO) "$(YELLOW)>>> Cleaning *.egg-info$(RESET)"
	$(FIND) . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
	$(UV_CACHE_DIR) -delete  2>/dev/null || true
	$(ECHO) "$(CYAN)>>> Done.$(RESET)"

# ------------------------------------------------------------
#  lint — standard type-checking and style enforcement
# ------------------------------------------------------------

lint:
	$(ECHO) ">>> flake8 …"
	flake8 .
	$(ECHO) ">>> mypy (standard) …"
	mypy . \
	    --warn-return-any \
	    --warn-unused-ignores \
	    --ignore-missing-imports \
	    --disallow-untyped-defs \
	    --check-untyped-defs

# ------------------------------------------------------------
#  lint-strict — maximum mypy strictness (recommended)
# ------------------------------------------------------------

lint-strict:
	$(ECHO) ">>> flake8 …"
	flake8 .
	$(ECHO) ">>> mypy (strict) …"
	mypy . --strict

