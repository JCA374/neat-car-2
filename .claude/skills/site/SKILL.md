---
name: site
description: Change, rebuild and check the project's GitHub Pages site in docs/ (overview, NEAT, PPO, race and championship pages, charts, ladder, network explorer, findings). Use when editing docs/*.html or docs/assets/*, changing what scripts/build_site.py exports, adding a chart or section, or checking how the page looks.
---

# The project page (`docs/`)

## What the page is for
It shows **what works, how we know, and the steps we took to learn it**, plus a **fun race**. Focus on the models (NEAT vs PPO), not on bug-fix history. Use plain short sentences with numbers, like the `outcome` notes.

## Map
| Page | Shows | Data (from `docs/assets/data/`) |
|---|---|---|
| `index.html` | champions, lap vs training-time scatter, leaderboard, findings | `leaderboard.json`, `findings.json` |
| `neat.html` | ladder, curves, network explorer, watch-it-think | `leaderboard.json`, `steps/<id>.json`, `think/<id>/<seed>.json`, `tracks.json` |
| `ppo.html` | ladder, curves | `leaderboard.json`, `steps/<id>.json` |
| `race.html` | ghost race of any steps | `drives/<id>.json`, `tracks.json` |
| `championship.html` | championship cars, two categories, exhibition races | `championship.json`, `champ/…`, `findings.json` |

Shared code: `assets/site.css` (colour tokens on `:root` + dark mode, `--neat`/`--ppo` colours), `assets/site.js` (helpers, `Site.ladder`), `assets/race-player.js` (race playback). Chart.js comes from the jsDelivr CDN, and there's no build step for HTML/JS.

## Rules
- **Never hand-edit `docs/assets/data/` or `docs/assets/gifs/`.** They come from `python scripts/build_site.py`, which reads `experiments/*/*.toml`, `experiments/findings.toml` and `outputs/*/*/{result,drives}.json`. To change the data, change `build_site.py` or its inputs and rebuild.
- Text that belongs to a step (`title`, `why`, `outcome`) lives in the step TOML. Findings live in `experiments/findings.toml`.
- Use the colour tokens from `site.css`, never raw hex in the pages. NEAT is always `--neat` (blue) and PPO `--ppo` (orange). Every change must work in light and dark mode and at phone width (no horizontal scroll).
- Charts: use the `dataviz` skill for chart choices and colours.
- Keep the pages static so GitHub Pages can serve `docs/` as-is.

## Steps
1. Make the change (HTML/JS/CSS, or `build_site.py` + rebuild).
2. `python scripts/build_site.py` and check that it printed no errors.
3. Serve locally: `python -m http.server 8000 -d docs` (in the background), then check http://localhost:8000/ with the browser tools if they're available (light + dark, about 375 px wide, console free of errors). If no browser is available, at least fetch each changed page and its JSON with `curl -s` and say the visual check is still open.
4. Stop the server afterwards.
5. Publishing is a separate step: use the `publish` skill, and only when the user asks.

## Done when
The site builds, the changed pages load without console errors in both themes, and the user knows what changed and what to look at.
