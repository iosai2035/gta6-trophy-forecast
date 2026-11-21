"""gta6-trophy-forecast: forecast Exophase earn rates for GTA VI (PS5) trophies.

Model (own design, standard library only)
------------------------------------------
The earn rate of a trophy is the share of Exophase-tracked players who have
earned it.  Two forces move it:

1. Dilution.  Tracked players arrive in launch waves (midnight unlocks around
   the world); every newcomer starts with nothing, so in the first hours
   rates fall even while the raw earn counts grow.  The house saw exactly
   this on launch night: 638 players tracked, every rate flat, then 3,868 at
   the first refresh with every rate but the 100% one lower.  For short
   horizons the forecast is therefore a dilution of the last read:
   last rate, divided by the modelled growth of players tracked, times
   (1 + earn growth) for the trophies new players can reach fastest.
2. Progression.  Players accumulate play hours.  Story-gated trophies need
   story progress, so they follow a threshold on play hours (chapter 1
   ~16h, the final mission ~72h, 100% completion ~200h).  Optional trophies
   instead follow a per-hour hazard: a player with H play hours has earned
   one with probability ceiling * (1 - exp(-H / tau)), where tau is the
   timescale on which eventual earners get round to it (a quick one-off
   ~20h, a 250-stunt grind ~150h, an unknown-method easter egg ~200h).
   Ceilings are the long-run share among Exophase-tracked (engaged)
   players, calibrated from the long-run analogues GTA V / Red Dead
   Redemption 2: first mission ~82-90%, final mission ~24-29%, 100%
   completion ~6%.  The median play hours of the tracked population grows
   slower than clock time, because newcomers keep joining young:
   H_eff = a * T**b with T the hours since the New Zealand unlock
   (2026-11-18T11:00Z), anchored at ~1.3h when T = 17.5h (first refresh)
   and ~16h when T = 181h (week one).  Play hours are lognormal across
   players (sigma 0.9), truncated at the physical maximum (hours since
   release).  The two regimes blend in logit space by horizon, with the
   dilution branch dominating at 90 minutes and progression at a week.

When the case history actually varies (the final case, with days of reads),
a per-trophy least-squares trend of logit(rate) against ln(1 + hours since
release) is blended with the prior in logit space, so real observed
progression overrides the prior where the data speak.

Quantiles are normal on the logit scale around the median, with a spread
that combines a random-walk term (0.5 * sqrt(days)), a per-class model
uncertainty, and a refresh-staleness term (Exophase refreshes on its own
schedule, so the answering read may reflect an earlier refresh).  The
first-mission trophy is special: every tracked player holds it (it stayed
at 100% through the first refresh), so it is forecast near 100% with a
tight spread.

I/O follows the starter's skeleton: read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR or /input), write <id>.out per case to
MAKEFUN_OUTPUT_DIR or /output, one row per trophy, five quantiles in
percent, non-decreasing, each 0.01..99.99.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# New Zealand unlock, the first moment anyone could play.
RELEASE = datetime(2026, 11, 18, 11, 0, 0, tzinfo=timezone.utc)

# Median play hours of the tracked population: H_eff = A * T**B, T in hours
# since release.  Anchors: 1.3h at T=17.5h (first refresh), 16h at T=181h.
_H_A = 0.0583
_H_B = 1.074
# Spread of log play hours across players.
_SIGMA_PLAY = 0.9
# Floor rate (percent): a trophy never quite reaches zero, a few
# pre-release players hold even the grinds.
_RATE_FLOOR = 0.02


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lg))))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Trophy profiles: (kind, param, long-run ceiling %, spread)
# kind "thresh": param is the play hours a player needs before the trophy
#   is reachable at all (story-gated: chapters, missions, 100%, platinum).
# kind "hazard": param is tau, the timescale in play hours on which
#   eventual earners get round to the trophy; P = ceiling*(1-exp(-H/tau)).
# Ceilings are the long-run share among Exophase-tracked (engaged) players,
# from GTA V / RDR2 long-run analogues.  Keyed by exact trophy name as
# printed by Exophase; unknown names fall back to the keyword rules in
# classify().
# ---------------------------------------------------------------------------
PROFILES = {
    "Vice Got a Price!": ("thresh", 200.0, 6.0, 0.9),
    "Just Like Paradise": ("thresh", 0.2, 99.9, 0.15),
    "Paradise Lost": ("thresh", 16.0, 88.0, 0.35),
    "T**s Ass Drug Money": ("thresh", 24.0, 78.0, 0.35),
    "Vice City, Baby!": ("thresh", 38.0, 68.0, 0.35),
    "Top Of The World, Ma": ("thresh", 54.0, 58.0, 0.40),
    "You Gotta Know When to Get Off": ("thresh", 72.0, 50.0, 0.40),
    "Trust": ("thresh", 20.0, 75.0, 0.40),
    "Break the Bank": ("thresh", 36.0, 65.0, 0.40),
    "Family Matters": ("thresh", 48.0, 55.0, 0.40),
    "DWNFALL": ("thresh", 52.0, 55.0, 0.40),
    "Hi-Jacked": ("thresh", 56.0, 55.0, 0.40),
    "Snow Storm": ("thresh", 68.0, 50.0, 0.40),
    "A Long Road": ("thresh", 200.0, 6.0, 0.90),
    "Cunning Stunts": ("hazard", 150.0, 14.0, 0.70),
    "Master of None": ("hazard", 30.0, 70.0, 0.50),
    "The Shootist": ("hazard", 200.0, 10.0, 0.80),
    "Top Fun": ("hazard", 80.0, 25.0, 0.60),
    "Earn Cash, Not Kudos": ("hazard", 30.0, 45.0, 0.50),
    "Baited My Tackle": ("hazard", 25.0, 40.0, 0.50),
    "Stick Fiddler": ("hazard", 20.0, 45.0, 0.50),
    "Road to Rehabilitation": ("hazard", 70.0, 32.0, 0.60),
    "This Was No Boat Accident": ("hazard", 70.0, 30.0, 0.60),
    "The Lone Rider": ("hazard", 70.0, 32.0, 0.60),
    "Strangers & Freaks": ("hazard", 90.0, 30.0, 0.60),
    "Supply & Demand": ("hazard", 80.0, 28.0, 0.60),
    "Spin On This!": ("hazard", 70.0, 30.0, 0.60),
    "Positive Mental Attitude": ("hazard", 40.0, 40.0, 0.60),
    "Fair Game": ("hazard", 80.0, 28.0, 0.70),
    "Don't Touch My Deck": ("hazard", 150.0, 18.0, 0.70),
    "You WAiNKer!": ("hazard", 60.0, 35.0, 0.60),
    "One Last Thing": ("hazard", 120.0, 20.0, 0.70),
    "Hot Together": ("hazard", 35.0, 45.0, 0.50),
    "Retail Therapy": ("hazard", 50.0, 35.0, 0.60),
    "Whip It Good": ("hazard", 35.0, 55.0, 0.50),
    "Gunslinger": ("hazard", 25.0, 50.0, 0.50),
    "Catch Me If You Can": ("hazard", 120.0, 55.0, 0.50),
    "Challenging Tank": ("hazard", 60.0, 35.0, 0.60),
    "Sneakthief": ("hazard", 50.0, 40.0, 0.60),
    "Maximum Withdrawal": ("hazard", 80.0, 25.0, 0.60),
    "A Classic Blitz Play": ("hazard", 50.0, 45.0, 0.50),
    "The Goodtime State": ("hazard", 45.0, 50.0, 0.50),
    "Pest Control": ("hazard", 70.0, 30.0, 0.60),
    "In The Beginning…": ("hazard", 200.0, 25.0, 0.90),
    "Capital Offense": ("hazard", 90.0, 30.0, 0.70),
    "Let’s Go Bowling": ("hazard", 60.0, 35.0, 0.60),
    "Personal Best": ("hazard", 90.0, 25.0, 0.70),
    "Actions Speak Louder": ("hazard", 120.0, 20.0, 0.70),
    "Bad Luck, I Guess": ("hazard", 200.0, 20.0, 0.90),
    "The Truth Is Out There": ("hazard", 200.0, 20.0, 0.90),
    "Ten Pin": ("hazard", 200.0, 25.0, 0.90),
    "Dime on the Dollar": ("hazard", 70.0, 30.0, 0.60),
    "Treat Yo Self": ("hazard", 30.0, 55.0, 0.50),
    "Little Boxes": ("hazard", 60.0, 35.0, 0.60),
    "On the Spot": ("hazard", 35.0, 45.0, 0.50),
}


def classify(trophy):
    """Return (kind, param, ceiling, class_sigma) for a trophy dict."""
    name = trophy.get("name", "")
    if name in PROFILES:
        return PROFILES[name]
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier", "")
    if "complete gone fishing" in desc:
        return ("thresh", 0.2, 99.9, 0.15)
    if "complete chapter 1" in desc:
        return ("thresh", 16.0, 88.0, 0.35)
    if "complete chapter" in desc or "final mission" in desc:
        return ("thresh", 60.0, 55.0, 0.40)
    if "complete" in desc and "mission" in desc:
        return ("thresh", 30.0, 55.0, 0.50)
    if "100% completion" in desc:
        return ("thresh", 200.0, 6.0, 0.90)
    if tier == "platinum":
        return ("thresh", 200.0, 6.0, 0.90)
    if tier == "gold":
        return ("thresh", 70.0, 40.0, 0.50)
    if tier == "silver":
        return ("hazard", 120.0, 20.0, 0.70)
    return ("hazard", 60.0, 35.0, 0.60)


def median_play_hours(target_time):
    """Median play hours of the tracked population at the target instant."""
    t_rel = max(0.5, (target_time - RELEASE).total_seconds() / 3600.0)
    return _H_A * (t_rel ** _H_B)


def prior_median(trophy, target_time):
    """Prior median earn rate (percent) at the target, from the time curve."""
    name = trophy.get("name", "")
    if name == "Just Like Paradise":
        return 99.99
    kind, param, ceiling, _sig = classify(trophy)
    t_rel = max(0.5, (target_time - RELEASE).total_seconds() / 3600.0)
    h_eff = median_play_hours(target_time)
    if kind == "hazard":
        # Optional trophy: per-hour hazard, evaluated at the population's
        # median play hours.  The hazard is slow because players only
        # occasionally attempt any one side activity.
        rate = ceiling * (1.0 - math.exp(-h_eff / max(1.0, param)))
        return max(_RATE_FLOOR, rate)
    # Story-gated trophy: share of players whose play hours exceed the
    # threshold, play hours lognormal(mu, SIGMA_PLAY) truncated at the
    # physical maximum (hours since release).
    hours = param
    mu = math.log(max(0.05, h_eff))
    h_max = max(t_rel, hours * 0.5)
    denom = phi((math.log(h_max) - mu) / _SIGMA_PLAY)
    if hours >= h_max:
        share = 0.0
    else:
        share = (denom - phi((math.log(hours) - mu) / _SIGMA_PLAY)) / max(denom, 1e-9)
    rate = ceiling * max(0.0, min(1.0, share))
    return max(_RATE_FLOOR, rate)


def players_growth_factor(case, target_time):
    """Modelled growth of players tracked between the last read and target.

    If the history shows players tracked growing, extrapolate its
    log-log trend (clipped); otherwise use the launch-wave shape: a big
    step while midnight unlocks are still rolling around the world, then
    slow steady growth.  Early on this is ~6x over 90 minutes (638 to
    3,868 at the first refresh); a week in it is a few-fold over days.
    """
    hist = case.get("history") or []
    last = hist[-1]
    n_last = last.get("players_tracked") or 0
    t_last = parse_time(last["at"])
    horizon_h = max(0.0, case["horizon_minutes"] / 60.0)
    t0 = max(0.5, (t_last - RELEASE).total_seconds() / 3600.0)
    # Data-driven: fit ln N against ln(1 + hours since release).
    pts = []
    seen = set()
    for read in hist:
        n = read.get("players_tracked")
        if not n:
            continue
        key = (read.get("slot"), n)
        if key in seen:
            continue
        seen.add(key)
        try:
            t = parse_time(read["at"])
        except Exception:
            continue
        pts.append((math.log1p(max(0.0, (t - RELEASE).total_seconds() / 3600.0)),
                    math.log(n)))
    if len(pts) >= 3 and n_last > 0:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 1e-9:
            slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
            intercept = my - slope * mx
            x_t = math.log1p(max(0.0, (target_time - RELEASE).total_seconds() / 3600.0))
            pred = math.exp(intercept + slope * x_t)
            g = pred / n_last
            if 1.0 < g:
                return min(8.0, max(1.2, g))
    # Launch-wave shape (no usable trend in the history).
    wave = math.exp(-t0 / 24.0)
    return 1.0 + 20.0 * wave * (1.0 - math.exp(-horizon_h / 2.0)) \
        + 0.02 * horizon_h * (1.0 - wave)


def earn_growth(trophy):
    """Extra earn-count growth over the horizon, by how reachable the
    trophy is for a brand-new player: quick one-offs grow fastest."""
    kind, param, _ceil, _sig = classify(trophy)
    if trophy.get("name") in ("Vice Got a Price!", "A Long Road"):
        return 0.0
    if kind == "thresh":
        return 0.25
    if param <= 30.0:
        return 0.8
    if param <= 80.0:
        return 0.4
    return 0.1


def dilution_median(case, idx, trophy, growth):
    """Short-horizon median: the last read's rate diluted by the growth
    of players tracked, plus the trophies' new earns.  None when the
    last read has no rate for this trophy."""
    last = case["history"][-1]
    rates = last.get("rates") or []
    if idx >= len(rates) or rates[idx] is None:
        return None
    if trophy.get("name") == "Just Like Paradise":
        return 99.99
    return max(_RATE_FLOOR, rates[idx] / growth * (1.0 + earn_growth(trophy)))


def trend_median(case, idx, target_time):
    """Least-squares trend of logit(rate) vs ln(1 + hours since release).

    Returns (median_percent, n_distinct) or (None, n) when the history
    carries no usable variation.
    """
    hist = case.get("history") or []
    pts = []
    seen = set()
    for read in hist:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        key = (read.get("players_tracked"), rates[idx])
        if key in seen:
            continue
        seen.add(key)
        try:
            t = parse_time(read["at"])
        except Exception:
            continue
        x = math.log1p(max(0.0, (t - RELEASE).total_seconds() / 3600.0))
        pts.append((x, logit(rates[idx])))
    n = len(pts)
    if n < 3:
        return None, n
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return None, n
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    x_t = math.log1p(max(0.0, (target_time - RELEASE).total_seconds() / 3600.0))
    pred = intercept + slope * x_t
    return clamp(inv_logit(pred)), n


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    target_time = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_h = case["horizon_minutes"] / 60.0
    walk = 0.5 * math.sqrt(max(0.0, horizon_days))
    growth = players_growth_factor(case, target_time)
    # Horizon blend: dilution dominates at 90 minutes, progression at a week.
    w_long = horizon_h / (horizon_h + 12.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        _kind, _param, _ceil, class_sig = classify(trophy)
        med = prior_median(trophy, target_time)
        dil = dilution_median(case, i, trophy, growth)
        if dil is not None:
            med = math.exp((1.0 - w_long) * math.log(dil) + w_long * math.log(med))
        # Blend with the observed trend where the history varies.
        trend, n_distinct = trend_median(case, i, target_time)
        if trend is not None:
            w = 0.5 * min(1.0, (n_distinct - 2) / 6.0)
            # Keep the trend from dragging the prior more than ~5x either way.
            lo = math.log(max(_RATE_FLOOR, med / 5.0))
            hi = math.log(min(99.9, med * 5.0))
            t_log = max(lo, min(hi, math.log(trend)))
            med = math.exp((1.0 - w) * math.log(med) + w * t_log)
        if trophy.get("name") == "Just Like Paradise":
            sigma = 0.4
            med = 99.99
        else:
            sigma = math.sqrt(walk * walk + class_sig * class_sig + 0.4 * 0.4)
        lg = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = clamp(inv_logit(lg + z * sigma))
            if prev is not None and q < prev:
                q = prev
            qs.append(round4(q))
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
