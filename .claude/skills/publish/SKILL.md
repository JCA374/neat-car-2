---
name: publish
description: Publish the CarRacing project - run tests, rebuild site data, commit and push so GitHub Pages (main branch, /docs) updates, then verify the live build. Use ONLY when the user explicitly says "publish" (or asks to push / deploy the page).
---

# Publish

Only run this when the user says so. Publishing changes the public page at https://jca374.github.io/neat-car-2/ (repo `JCA374/neat-car-2`, Pages serves **`main` /docs**).

## Steps
1. **Check nothing is training** (`pgrep -af "run_step|run_queue|ppo_lab|carracing_.*train"`). If something is, a step's outputs may be half-written, so ask whether to wait.
2. **Tests:** `pytest tests/ -q`. If any fail, stop and show the output.
3. **Rebuild:** `python scripts/build_site.py`.
4. **Review what goes out:** `git status` and `git diff --stat`.
   - Tracked: code, `experiments/`, `docs/` (including the generated data and GIFs), and `outputs/<model>/<id>/{model.*,result.json,drives*.json,lap*.gif,neat.cfg,exhibition_*.json}`.
   - Never commit: `checkpoints/`, `logs/`, `outputs/ppo/lab/` trial dirs, `.venv`, or anything large that isn't one of the above (check sizes with `du -sh`).
   - Show the user the list if it holds anything surprising.
5. **Commit:** a message in the style of `git log --oneline` (what changed and the headline result, e.g. "PPO p14: raw3 every 4 frames, 4/5 at 21.7 s"). End it with the attribution line from the session instructions.
6. **Branch:** Pages builds from `main`. If the current branch isn't `main`, ask the user: merge into `main` (fast-forward if possible) and push, or push the branch only (the page won't change). Never force-push.
7. **Push**, then **verify**:
   ```bash
   gh api repos/JCA374/neat-car-2/pages/builds/latest --jq '{status, commit, error: .error.message}'
   ```
   Wait for `built` (check every 30–60 s with Monitor, not a sleep loop), and confirm that `commit` matches the pushed head. Then fetch https://jca374.github.io/neat-car-2/ and one changed data file (for example `assets/data/leaderboard.json`) and check that the new content is there.

## Done when
The push succeeded, the Pages build says `built` for our commit, and the live page shows the new data. Report the commit hash and the link.
