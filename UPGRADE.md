# Upgrading the NEM Price Forecaster Add-on

This document describes how to safely roll forward to a new commit of
`purcell-lab/nem-price-forecaster` (or upstream `BrettLynch123/nem-price-forecaster`)
while preserving the calibration history that the sidecar accumulates on disk.

It is the result of repeated operational surprises during development — most
notably that "rebuild" can quietly run the old commit, and that the right
recovery path is rarely the most aggressive one.

## TL;DR — which Supervisor action should I use?

```
configuration change only ............... restart
new commit available, normal upgrade .... update            ← default
container looks wedged, code unchanged .. rebuild
repo URL changed or supervisor confused . uninstall + install
force-push rewrote main ................. remove_repository + add_repository
                                          (then re-install)
```

Always confirm afterwards with:

```
GET http://<addon-host>:8765/version
```

The `git_sha` field tells you which commit is actually running in the
container — independent of what the Supervisor UI claims is installed.

## What persists across each action?

| State                                          | restart | update | rebuild | uninstall+install | remove_repo+add_repo |
| ---------------------------------------------- | :-----: | :----: | :-----: | :---------------: | :------------------: |
| Calibration observations (`calibration_*.json`) |   ✅    |   ✅   |   ✅    |        ✅¹        |          ✅¹         |
| Load observations (`load_obs_*.json`)           |   ✅    |   ✅   |   ✅    |        ✅¹        |          ✅¹         |
| Add-on options (region, lat/lon, etc.)          |   ✅    |   ✅   |   ✅    |        ❌²        |          ❌²         |
| Supervisor git clone (`/data/apps/git/...`)     |   ✅    |   🔄   |   ✅³   |        ✅         |          🔄          |
| Built container image                           |   ✅    |   🔄   |   🔄    |        🔄         |          🔄          |

✅ preserved · 🔄 replaced · ❌ reset to schema defaults

¹ Stored under the host's `/share/nem_forecaster_data/`, which is bind-mounted
into the container. The Supervisor's add-on lifecycle does not touch `/share`.

² Add-on options live in Supervisor state, not in `/share`. Re-installing
returns them to the defaults declared in `config.yaml`. Either record them
elsewhere before uninstalling, or take an HA backup first.

³ `rebuild` reuses the cached git clone. If the clone is stale (e.g. a force-
push happened), `rebuild` will quietly rebuild the old commit. This is the
single most common upgrade footgun. Use `update` or `remove_repository` +
`add_repository` instead when in doubt.

## Recommended upgrade procedure

1. **Snapshot the current state.**
   ```
   GET /version       → record git_sha
   GET /health        → record observation counts
   ```
   Optional: take a Home Assistant backup so add-on options can be restored
   if anything goes wrong.

2. **Run `update`** (Supervisor → Add-ons → NEM Price Forecaster → Update).
   This is the default path. It pulls the latest commit from the registered
   repository, rebuilds the image, and restarts the container. Add-on
   options and `/share` are both preserved.

3. **Verify.**
   ```
   GET /version       → git_sha matches the new commit?  observation count
                        matches step 1?  build_time_utc is recent?
   GET /health        → price_forecast_ready is true?
   ```

4. If `git_sha` did not change but you expected it to, the Supervisor served
   from a cached clone. Fall back to the **clean-clone procedure** below.

## Clean-clone procedure (when `update` is not enough)

This is the right path when:

- A force-push rewrote `main` (`git_sha` from `/version` doesn't match what
  GitHub shows on `main`).
- The repository URL changed (e.g. fork rename, ownership transfer).
- The Supervisor consistently rebuilds an old commit and won't move forward.

Steps:

1. Record the current add-on options and observation counts (see step 1
   above) — uninstall resets the options.
2. Stop and uninstall the add-on:
   `ha_manage_addon(slug=..., action="stop")` then
   `ha_manage_addon(slug=..., action="uninstall")`.
3. Remove the repository from the store:
   `ha_manage_addon(action="remove_repository", repository=<slug>)`.
4. Add it back:
   `ha_manage_addon(action="add_repository", repository="https://github.com/purcell-lab/nem-price-forecaster")`.
5. Install and start:
   `ha_manage_addon(slug=..., action="install")` then `action="start"`.
6. Restore the add-on options that were recorded in step 1.
7. Verify with `GET /version` and `GET /health`. The calibration observations
   should be intact because `/share/nem_forecaster_data/` is never touched
   by these operations.

## What the `/version` endpoint reports

```json
{
  "git_sha": "de9965f...",
  "git_branch": "main",
  "build_time_utc": "2026-06-17T06:42:00Z",
  "addon_version": "0.2.0",
  "build_arch": "amd64",
  "api_version": "1.0.0",
  "python_version": "3.11.x",
  "data_dir": "/nem_forecaster_data",
  "region": "QLD1",
  "persisted_files": [
    {
      "path": "calibration_QLD1.json",
      "size_bytes": 247829,
      "modified_utc": "2026-06-17T06:21:37Z"
    },
    {
      "path": "load_obs_QLD1.json",
      "size_bytes": 12450,
      "modified_utc": "2026-06-17T06:21:37Z"
    }
  ]
}
```

The `git_sha`, `git_branch`, `build_time_utc`, `addon_version`, and
`build_arch` fields are baked into the image at build time via Dockerfile
`ARG`s. The HA Supervisor's builder injects `BUILD_REF` (git SHA),
`BUILD_DATE`, `BUILD_VERSION`, and `BUILD_ARCH` automatically. The
standalone sidecar Dockerfile accepts `GIT_SHA`, `GIT_BRANCH`, and
`BUILD_TIME` as build args; pass them explicitly when building locally:

```bash
docker build \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
  --build-arg BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t nem-price-forecaster-sidecar sidecar/
```

When any field is `"unknown"`, the image was built without that argument
(typically a local dev build) and the version endpoint cannot identify the
commit.
