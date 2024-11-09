#!/bin/bash

pre-commit run --config pre-commit/beautify.yaml --files $(find autoop app -type f -name "*.py")
