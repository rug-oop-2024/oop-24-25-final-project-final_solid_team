#!/bin/bash

pre-commit run --config pre-commit/beautify.yaml --files $(find autoop -type f -name "*.py")
