# Reports

## Purpose

Report helpers produce compact summaries from training and evaluation
artifacts.

## Contents

- [generate_report.py](generate_report.py): report CLI.
- [report_helpers.py](report_helpers.py): shared loading and formatting
  helpers.
- [report_render.py](report_render.py): rendering helpers.

## Rules

- Read artifacts from configured data directories.
- Keep generated report outputs out of source unless curated as docs.
