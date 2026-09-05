# Changelog — Edge

This is the **queue**, not a release history. Entries accumulate under
`## Unreleased` as work lands on edge; `scripts/promote.sh` is the moment they
move into `../occupancy-forecast/CHANGELOG.md` under a stable version number and
this section is emptied again.

File each entry under the same `### Added` / `### Changed` / `### Fixed` /
`### Removed` headings the stable changelog uses, so that a promotion is a move
rather than a rewrite.

## Unreleased

### Fixed

- Installing the add-on from the repository URL failed to build. The Ingress
  panel's compiled bundle was gitignored, so Supervisor's clone contained no
  `panel/dist` and the image build died at the `COPY` with a Docker checksum
  error that named neither the panel nor the cause. The bundle is now committed,
  which is the only way an add-on that is built from a clone and compiles no
  frontend on the box can have one.

### Added

- `scripts/check-panel.sh` and `scripts/panel-source-hash.sh`, and a stamp file
  `panel/dist/.source-hash` written by `scripts/build-panel.sh`. A committed
  artifact can be stale, and a stale panel is silent — it installs cleanly and
  serves old code. The pre-commit hook now refuses a panel source edit whose
  bundle was not rebuilt, and `scripts/deploy-stable.sh` checks the stamp instead
  of rebuilding (`scripts/promote.sh` builds stable's bundle now, so that it is
  part of the promotion commit).
