#!/bin/bash
pre-commit run --files $(find app autoop -type f -name "*.py")

# python3 tests/test_game.py <- Does not work at the moment
python3 -m autoop.tests.main
