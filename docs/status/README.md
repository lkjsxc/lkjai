# Status Index

Owner: `docs/status/README.md`.
State: canonical gap map.

Use this subtree for current implementation gaps and acceptance evidence
boundaries. Contract owners remain in `docs/architecture`, `docs/product`, and
`docs/operations`; these files point to the active blockers.

## Read This Section When

- You need the current gap list before changing behavior.
- You need to distinguish accepted evidence from diagnostics.
- You need route transcript and runtime transcript boundaries.

## Child Index

- [implementation-gaps.md](implementation-gaps.md): active code gaps by lane.
- [route-evidence.md](route-evidence.md): accepted route transcript rules.
- [runtime-tools.md](runtime-tools.md): active tool profiles and memory status.
- [decoder-acceptance-blockers.md](decoder-acceptance-blockers.md): decoder
  promotion blockers.

## Stop Condition

Remove a status item only after the owning contract, implementation, tests, and
verification evidence agree. Do not remove a blocker because a sidecar, fixture,
or dry-run reports the accepted field names.
