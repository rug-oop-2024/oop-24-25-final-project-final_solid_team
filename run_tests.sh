#!/bin/bash
pre-commit run --config pre-commit/strict.yaml --files $(find autoop/core autoop/functional -type f -name "*.py")

# python3 tests/test_game.py <- Does not work at the moment
python3 -m autoop.tests.main
