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

**That is the fact to hold on to, because it cuts both ways.** It is what keeps
edge and stable apart — `..._occupancy_forecast` and
`..._occupancy_forecast_edge` resolve to different topic roots, so the two never
touch. It is *also* why the repository prefix is invisible, and Supervisor has
two of those:

| how it was installed | example slug | resolves to |
|---|---|---|
| from this repository's URL | `28b1f84a_occupancy_forecast` | `occupancy_forecast` |
| copied into `/addons/` | `local_occupancy_forecast` | `occupancy_forecast` |

So each of the two add-ons can be installed **either way** — four installs are
possible on one box — and they form **two colliding pairs**. The store copy and
the local copy of the *same* add-on share a topic root and an MQTT client id.
Nothing separates them, and the failure is silent: the broker hands a duplicated
id to whoever connected last and disconnects the other, forever, with nothing in
any log. No error path catches it, because both instances read their slug from
Supervisor perfectly well and simply agree on the answer.

**Run one of each pair.** On the maintainer's box that is edge local and stable
from the store, which is why the two have different deploy triggers:

| add-on | slug there | how it updates |
|---|---|---|
| edge | `local_occupancy_forecast_edge` | `scripts/deploy-edge.sh` — rsync + rebuild. No commit, no push, no version bump: a local add-on rebuilds from its directory. |
| stable | `28b1f84a_occupancy_forecast` | bump `occupancy-forecast/config.yaml`'s `version:`, commit, **push**. Supervisor then offers the update, exactly as it does for anyone else who added this repository. |

That pairing is a choice, not a law — but whichever way round you run it, adding
the second copy of an add-on you already have installed is the mistake to avoid,
and it is why there is no deploy script for stable.

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
An edit made there survives only until the next promotion, which then destroys it
silently.

**Nothing enforces this.** There was a `.githooks/pre-commit` that refused commits
touching `occupancy-forecast/` without `PROMOTE=1`; it was removed deliberately,
because it assumed the edge work and the promotion were two separate commits and
they are not (see the routine below). So the rule is prose, and prose is all there
is: an edit made directly in the stable tree will be committed without complaint
and destroyed by the next promotion.

## The panel

The Ingress panel is a React + TypeScript app in `occupancy-forecast-edge/panel/`.
`occupancy_forecast/web/` no longer renders it; it only serves the build and
substitutes the add-on's name into the title.

**It is never built on the Home Assistant box.** `scripts/build-panel.sh` runs
Vite in a container here, so `panel/dist/` travels with the Python and the
Dockerfile only `COPY`s it. Every other add-on on that box reports `build=False`
— a prebuilt image somebody else compiled — and adding a node stage to the
Dockerfile would make this the only thing on the machine compiling a frontend
locally.

**`dist/` is committed. Only `node_modules/` is gitignored.** This repository is
a valid add-on repository, and an add-on installed from a repository URL is a git
clone and nothing more — so an ignored bundle is an add-on that cannot be
installed from the store at all, failing at the `COPY` with a Docker checksum
error that names neither the panel nor the cause. That is what it did.

A committed artifact can be stale, and a stale panel is silent: it installs
cleanly and serves old code. So `build-panel.sh` writes `panel/dist/.source-hash`
(a hash of `src/`, `index.html`, `package*.json`, `tsconfig.json`,
`vite.config.ts`) and `scripts/check-panel.sh` compares it back. `scripts/test.sh`
runs that check, so a green suite means the committed bundle matches its source
— and a suite that fails there is telling you to run
`scripts/build-panel.sh occupancy-forecast-edge` and test again. `promote.sh`
rebuilds both bundles itself — edge's before the rsync, stable's after — so the
pair is fresh by construction at promotion.

`promote.sh` copies the panel's source and not edge's build output — it runs
`build-panel.sh occupancy-forecast` afterwards instead, so stable's bundle is
compiled from stable's own tree *and* lands in the promotion commit.

The UI has no runtime tests. `tsc --noEmit` runs in `scripts/test.sh`, and
`panel/src/types.ts` plus `occupancy_forecast/tests/test_api_contract.py` are two
halves of one contract that nothing checks automatically: **a renamed API field
has to change in both.**

## The routine for any change

1. Edit under `occupancy-forecast-edge/`.
2. If the edit touched `panel/`, `scripts/build-panel.sh occupancy-forecast-edge`
   first: the bundle is committed alongside its source. Then `scripts/test.sh`
   — must pass. It checks the bundle against the source, so a stale bundle
   fails the suite rather than shipping silently.
3. Add a line to `occupancy-forecast-edge/CHANGELOG.md` under `## Unreleased`,
   filed under `### Added`, `### Changed` or `### Fixed`. That section is the
   queue; at promotion the block is copied verbatim into
   `occupancy-forecast/CHANGELOG.md`, which is Keep a Changelog and carries the
   versioning rules in its preamble.

   **Write it for somebody running the add-on, not for yourself.** One to three
   plain sentences saying what changed on their side — a copied block lands
   unedited in the Changelog tab that every store user sees, so the register you
   write in is the register they read. Name entities, options and panel pages
   rather than modules and functions, and keep anything they have to act on. The
   mechanism, the measurement and the alternative you rejected are worth
   recording, but they go in the **commit message**, where the reader is
   somebody reading the diff.
4. `scripts/deploy-edge.sh` — rsyncs to the HA box and rebuilds the add-on.
   **Deploying edge needs no version bump and no push**: edge is a *local* add-on,
   so `ha addons rebuild` is the trigger. The script stamps the deployed copy with
   the commit sha so the add-on page says which build is running. (Stable is not
   local and does not work this way — see the top of this file.)

## Pushing edge work, and promoting to stable

These are two different things, and the difference is what lets a change soak.

**Pushing edge work does not deploy stable.** Stable's only trigger is a changed
`version:` in `occupancy-forecast/config.yaml`. So edge work can be committed and
pushed on its own — as often as you like, and however many commits it takes — and
store users see nothing, as long as the push leaves the generated
`occupancy-forecast/` tree and its version alone. That is the normal way to let a
change run on edge for a while before deciding it has earned stable.

Two things to get right on such a push: if the work touched `panel/`, run
`scripts/build-panel.sh occupancy-forecast-edge` first (`scripts/test.sh` will
refuse a stale bundle, so a green run is the proof); and keep adding to the
`## Unreleased` block in `occupancy-forecast-edge/CHANGELOG.md`, which accumulates
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
prints the `diff` that checks it.

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

`scripts/test.sh` runs `tsc --noEmit`, then `check-panel.sh`, then pytest in
`python:3.13-slim` with the pinned dependencies the add-on ships. Five
hundred-odd tests: no network, no Home Assistant, no broker. Dev-only pins go in
`requirements-dev.txt` — the shipped image deliberately carries no test
framework.

`test.sh`, `build-panel.sh`, `check-panel.sh` and `promote.sh` work anywhere
Docker does. `deploy-edge.sh` and `backfill-store-from-influx.sh` are
**author-local**: they rsync to `HOST=ha`, an ssh alias for one particular box,
so they do nothing useful in a fresh clone.

Versions are documentation rather than a mechanism: stable is semver and moves
only on a promotion, edge is `<next-stable>-dev`. The stable changelog's own
preamble — not this file — says what a MAJOR, MINOR or PATCH bump means here.

The local demo instance, and the only supported way to screenshot the panel:
[`docs/demo-instance.md`](docs/demo-instance.md).

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
  switch it off. That boundary is the whole blast radius and should stay where it
  is.
