# Working in this repository

Two add-ons ship from here — `occupancy-forecast/` (stable) and
`occupancy-forecast-edge/` (development). They are the same code at different
release points, and they are designed to run **side by side** so a change can be
compared against what you already trust.

Every identity string derives from the add-on's slug, via
`config.resolve_topic_prefix()`, which drops the repository prefix:

```python
_, _, name = slug.partition("_")     # first underscore only
```

That keeps edge and stable apart, and it also makes the repository prefix
invisible, of which Supervisor has two:

| how it was installed | example slug | resolves to |
|---|---|---|
| from this repository's URL | `28b1f84a_occupancy_forecast` | `occupancy_forecast` |
| copied into `/addons/` | `local_occupancy_forecast` | `occupancy_forecast` |

So the store copy and the local copy of the *same* add-on share a topic root and
an MQTT client id, and the broker silently keeps whichever connected last.
**Never install both copies of one add-on.** On the maintainer's box edge is
local and stable comes from the store, which is why the two have different
deploy triggers and why there is no deploy script for stable:

| add-on | slug there | how it updates |
|---|---|---|
| edge | `local_occupancy_forecast_edge` | `scripts/deploy-edge.sh` — rsync + rebuild. No commit, no push, no version bump: a local add-on rebuilds from its directory. |
| stable | `28b1f84a_occupancy_forecast` | bump `occupancy-forecast/config.yaml`'s `version:`, commit, **push**. Supervisor then offers the update, exactly as it does for anyone else who added this repository. |

One name deliberately did **not** follow the rename: the `/data` row keys
`occupancy_ml.collector` and `occupancy_ml.{slug}_distance`. They are data, not
code — the archive is full of rows already written under them and Home Assistant
cannot re-supply it, so renaming them would make the add-on stop reading its own
history. They now match nothing else in the tree, which is exactly why
`STORE_LOCAL_NAMES` in `test_portability.py` pins them with a note.

## Where code lives

- `occupancy-forecast-edge/` — **the source of truth. Edit here.**
- `occupancy-forecast/` — **generated. Never edit any file here** except
  `config.yaml`, `DOCS.md` and `CHANGELOG.md`, and only during a promotion. The
  exception is correcting `CHANGELOG.md` in place — a wrong or badly formatted
  entry can be fixed whenever it is spotted.

`occupancy-forecast/occupancy_forecast/*.py` is produced by `scripts/promote.sh`.
Nothing enforces this: an edit made in the stable tree commits without complaint
and the next promotion destroys it silently.

## The panel

The Ingress panel is a React + TypeScript app in `occupancy-forecast-edge/panel/`.
`occupancy_forecast/web/` only serves the build and substitutes the add-on's name
into the title.

**It is never built on the Home Assistant box.** `scripts/build-panel.sh` runs
Vite in a container here and the Dockerfile only `COPY`s `panel/dist/`.

**`dist/` is committed. Only `node_modules/` is gitignored.** An add-on installed
from a repository URL is a git clone and nothing more, so an ignored bundle fails
the `COPY` with a Docker checksum error that names neither the panel nor the
cause.

A committed artifact can be stale, and a stale panel is silent: it installs
cleanly and serves old code. So `build-panel.sh` writes `panel/dist/.source-hash`
(a hash of `src/`, `index.html`, `package*.json`, `tsconfig.json`,
`vite.config.ts`) and `scripts/check-panel.sh` compares it back. `scripts/test.sh`
runs that check, so a green suite means the committed bundle matches its source
— and a suite that fails there is telling you to run
`scripts/build-panel.sh occupancy-forecast-edge` and test again. `promote.sh`
rebuilds both bundles itself — edge's before the rsync, stable's after — so the
pair is fresh by construction at promotion, and stable's bundle is compiled from
stable's own tree.

The UI has no runtime tests. `tsc --noEmit` runs in `scripts/test.sh`, and
`panel/src/types.ts` plus `occupancy_forecast/tests/test_api_contract.py` are two
halves of one contract that nothing checks automatically: **a renamed API field
has to change in both.**

## The routine for any change

1. Edit under `occupancy-forecast-edge/`.
2. If the edit touched `panel/`, `scripts/build-panel.sh occupancy-forecast-edge`
   first. Then `scripts/test.sh` — must pass.
3. Add a line to `occupancy-forecast-edge/CHANGELOG.md` under `## Unreleased`,
   filed under `### Added`, `### Changed`, `### Fixed` or `### Removed` — the
   same headings the stable changelog uses. That section is the queue; at
   promotion the block is copied verbatim into `occupancy-forecast/CHANGELOG.md`.
   Both files carry **no preamble and no title**: line 1 is a version heading,
   because the store's Changelog tab shows the whole file to every user. The
   versioning rules are in [Tests and scripts](#tests-and-scripts) below, not in
   the changelogs.

   **Write it for somebody running the add-on, not for yourself.** One to three
   plain sentences saying what changed on their side — a copied block lands
   unedited in the Changelog tab that every store user sees, so the register you
   write in is the register they read. Name entities, options and panel pages
   rather than modules and functions, and keep anything they have to act on. The
   mechanism, the measurement and the alternative you rejected are worth
   recording, but they go in the **commit message**, where the reader is
   somebody reading the diff.
4. `scripts/deploy-edge.sh` — rsyncs to the HA box and rebuilds the add-on,
   stamping the deployed copy with the commit sha so the add-on page says which
   build is running.

## Pushing edge work, and promoting to stable

These are two different things, and the difference is what lets a change soak.

**Pushing edge work does not deploy stable.** Stable's only trigger is a changed
`version:` in `occupancy-forecast/config.yaml`, so edge work can be committed and
pushed as often as you like and store users see nothing, as long as the push
leaves the generated `occupancy-forecast/` tree and its version alone. The
`## Unreleased` block in `occupancy-forecast-edge/CHANGELOG.md` accumulates
across as many pushes as the soak takes and crosses over in one piece at
promotion.

**The promotion itself is ONE commit.**

```sh
# edge is happy and has soaked long enough
scripts/promote.sh [--no-test]     # rsync edge -> stable, rebuild both bundles
# bump occupancy-forecast/config.yaml's version:, retitle and copy ## Unreleased
git add -A && git commit           # ONE commit: generated tree, version, changelog
git push                           # THIS is the stable deploy
```

The changelog step is a **copy, not a move**: give edge's `## Unreleased` block
the version heading, open a fresh empty `## Unreleased` above it, and copy the
retitled block into stable. Edge keeps its own copy, so its changelog is the
whole record of what edge has run rather than only what is still queued. The two
files must be identical from the first version heading down, and `promote.sh`
prints the `diff` command that checks it.

Keep the generated tree, the version bump and the changelog together in the one
commit. Split across commits they describe a stable add-on that never existed at
any commit, and the history stops being able to say what stable was running.

`promote.sh` promotes the **working tree**, not a commit, so it neither requires a
clean tree nor cares whether edge was committed first — which is exactly why the
soak above is free. `--no-test` skips `scripts/test.sh` and is honest only when it
has just been run against these exact files.

`git add -A`, not `git commit -a`: the rebuilt panel bundles can contain new files
and `-a` will not pick those up.

## Tests and scripts

`scripts/test.sh` runs `tsc --noEmit`, then `check-panel.sh`, then pytest with
the pinned dependencies the add-on ships. It takes a few minutes: no network, no
Home Assistant, no broker. Dev-only pins go in `requirements-dev.txt` — the
shipped image deliberately carries no test framework.

pytest runs in `occupancy-forecast-test:<hash>`, tagged from both requirements
files, so a moved pin cannot be served a stale image.

**Every script here that runs a container sources `scripts/container-guard.sh`,
and it is not optional.** A script killed mid-run leaves its container behind,
and three orphaned containers once deadlocked this box. The guard names, labels
and locks every container (a second run refuses, it does not queue) and gives it
an in-container `timeout`, the only mechanism that survives a `kill -9` of the
shell. `TEST_TIMEOUT` (seconds, default 900) sets it for the suite.

`docker ps --filter label=hass-occupancy.script` shows what a run has open.

**`scripts/check-pins.sh` — run it whenever a numerical pin moves.** It
installs and runs the wheels on amd64, and disassembles the aarch64 ones, since
nothing here is an ARM machine: it fails when an object gains ARMv8.1 LSE
atomics outside libgcc's dispatch, the pyarrow 21.0.0 bug that aborted on a
Pi 4. `test.sh` cannot cover this — it pins no `--platform`, so it tests
whichever architecture the developer's machine is. The aarch64 half is a diff
against `scripts/arm-baseline.json`; `--update-baseline` is honest only once the
pin has run on a real Pi.

**`scripts/build-image.sh <tree>` then `scripts/smoke-image.sh <tree>`** build
the real add-on image (`BUILD_FROM` read from that tree's `build.yaml`), run the
pinned stack inside it, and boot the server until `/health` answers with the
`code.fingerprint` of the tree it was built from. `check-pins.sh` tests the
pins; this tests the Dockerfile — the 0.2.0 class, which would not start.

`.github/workflows/ci.yml` runs the suite and both scripts on amd64 and aarch64.
Two advisory jobs run only when a pin, the `Dockerfile` or `build.yaml` moves:
`pins-arm` (`check-pins.sh --only arm`) and `cortex-a72`, the stack under
qemu-user modelling a Pi 4. **A green arm64 runner does not clear a Pi 4** —
those cores have LSE.

`test.sh`, `build-panel.sh`, `check-panel.sh`, `check-pins.sh`,
`build-image.sh`, `smoke-image.sh` and `promote.sh` work on any Linux with
Docker. `deploy-edge.sh` and
`backfill-store-from-influx.sh` are **author-local**: they rsync to `HOST=ha`, an
ssh alias for one particular box, so they do nothing useful in a fresh clone.

Versions are documentation rather than a mechanism: stable is semver and moves
only on a promotion, edge is `<next-stable>-dev`. Both changelogs are
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). What a bump means:

- **MAJOR** — something the user has to act on: a renamed or removed MQTT topic,
  entity or `unique_id`; a removed or renamed option in `config.yaml`; a
  `/data/history.db` migration you cannot roll back from.
- **MINOR** — additive: new published entities, options, endpoints, or a new
  signal the forecast can use.
- **PATCH** — fixes and internals, with no change to any surface above.

The local demo instance, and the only supported way to screenshot the panel:
[`docs/demo-instance.md`](docs/demo-instance.md).

## Comments

Comments say **why**, and the code says what.

The budget, and it is a real budget:

- **File or module header: up to about five lines.** What it is, when to run it
  or what calls it, and the one trap that would cost somebody an afternoon.
- **Inline: one line.** Two if the reason genuinely needs a second.
- **If the comment is longer than the code it sits above, it is too long.**

What does *not* belong in a comment: the measurements you took, the alternative
you rejected, the bug you hit on the way, or a narration of how the code came to
look like this. Those go in the **commit message** — the same rule the changelog
section above states, for the same reason. A reader of the diff wants them; a
reader of the file next year does not.

An incident that shaped the code is worth one sentence naming it, not a
retelling.

This applies to prose in docs too, agents included. Prefer cutting to adding.

## Never

- **Never hardcode an identity string.** Anything that names an entity, topic,
  notification or log line derives from `config.topic_prefix()` or
  `config.display_name()`. A hardcoded one is the `NOTIFY_COLLECTING` bug again.
  Changing `topic_prefix()`, `display_name()`, `state_prefix()`, `unique_id`
  construction, MQTT client ids or discovery topics (`config.py`, `predict.py`,
  `server.py`) is how the two add-ons stop keeping out of each other's way, and
  the failure is the silent client-id collision described at the top.
- **Never rewrite `/data/history.db`'s schema — migrate it.** That archive is
  unrecoverable: Home Assistant's recorder keeps ~10 days and cannot re-supply it.
- Never write a real entity id, person name, token, IP or hostname into code,
  tests or docs. The tests run against a synthetic household on purpose: that is
  what stops one particular installation creeping back into the package.
- Never commit an Influx token or MQTT password. Supervisor injects them at
  runtime via `run.sh`.
- Never add a test dependency to `requirements.txt`. Dev-only pins go in
  `requirements-dev.txt`, because the shipped image deliberately carries no test
  framework.
- Never add a file that only one add-on needs. `promote.sh` and `deploy-edge.sh`
  run `rsync --delete` against an exclude list, so a file present in only one tree
  is removed on the next run.
- Never let the add-on write to Home Assistant beyond a persistent notification.
  This is advisory only. Acting on the forecast — taking a house out of heating
  setback, say — belongs in the user's own automations, where they can see it and
  switch it off.
