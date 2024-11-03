#!/bin/bash

pre-commit run --config pre-commit/beautify.yaml --files $(find . -type f -name "*.py")
