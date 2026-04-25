#!/usr/bin/env python3
import sys

if "--help" in sys.argv:
    print("Usage: kimi --quiet --print --final-message-only -p --input-format --output-format")
else:
    print("I will explain instead of returning JSONL.")
