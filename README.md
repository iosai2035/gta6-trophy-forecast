# gta6-trophy-forecast
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/iosai2035/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/iosai2035/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at the case target, as five quantiles (5/25/50/75/95%) on the logit scale.

## Method

Two forces move an Exophase earn rate: **dilution** — tracked players arrive in launch waves, every newcomer starts with nothing, so rates fall in the first hours even as earn counts grow (the house saw 638 players tracked, then 3,868 at the first refresh with every rate but the 100% one lower) — and **progression** — players accumulate play hours. Story-gated trophies follow a threshold on play hours (chapter 1 ~16h, the final mission ~72h, 100% completion ~200h); optional trophies follow a per-hour hazard, `ceiling * (1 - exp(-H/tau))`, with long-run ceilings calibrated from the GTA V / Red Dead Redemption 2 analogues (first mission ~82–90%, final mission ~24–29%, 100% completion ~6%). Median play hours of the tracked population grow slower than clock time because newcomers keep joining young. The two regimes blend in logit space by horizon (dilution dominates at 90 minutes, progression at a week), and when a case's history actually varies, a per-trophy logit trend against log time since release is blended in. Quantiles are normal on the logit scale with a spread combining a random-walk term, per-class model uncertainty, and Exophase refresh staleness. The first-mission trophy is forecast at 99.99%: every tracked player holds it.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` per case: one row per trophy, five non-decreasing quantiles in percent.

<sub>Entered on [make.fun](https://make.fun/?ref=57z043hx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
