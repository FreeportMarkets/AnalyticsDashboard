"""HL-sourced perp volume — 1:1 with Hyperliquid per-fill data.

Volume is computed from HL `userFillsByTime` (every fill's price * size), the
same per-fill notional HL charges builder fees on. This is exact, unlike
reconstructing from our trade log (which stores intended order size, not filled
size, so it drifts ~15% on partial/IOC fills).

Only PERP fills count (open/close, long/short). Spot fills (dir "Buy"/"Sell")
and "Settlement" are excluded. Fills are deduped by trade id (`tid`).
"""
import json
import urllib.request

PERP_DIRS = frozenset({"Open Long", "Close Long", "Open Short", "Close Short"})
_HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def fill_notional(fill):
    """One fill's notional = |size| * price. Always positive."""
    return abs(float(fill["sz"])) * float(fill["px"])


def is_perp_fill(fill):
    """True only for the four perp directions; excludes spot + settlement."""
    return fill.get("dir") in PERP_DIRS


def perp_volume_from_fills(fills):
    """Sum |size|*price over perp fills, deduped by `tid`."""
    seen = set()
    total = 0.0
    for f in fills:
        tid = f.get("tid")
        if tid is not None:
            if tid in seen:
                continue
            seen.add(tid)
        if is_perp_fill(f):
            total += fill_notional(f)
    return total


def compute_perp_volume(wallets, start_ms, end_ms, fetcher):
    """Fetch each wallet's fills, keep those in [start_ms, end_ms), and sum the
    perp notional. `fetcher(wallet, start_ms)` returns a list of HL fill dicts —
    injected so this is testable without network and swappable for caching.
    """
    seen = set()
    deduped = []
    for wallet in wallets:
        for f in fetcher(wallet, start_ms):
            t = f.get("time")
            if t is not None and not (start_ms <= int(t) < end_ms):
                continue
            tid = f.get("tid")
            if tid is not None:
                if tid in seen:
                    continue
                seen.add(tid)
            deduped.append(f)
    return perp_volume_from_fills(deduped)


# --- real network fetcher (used by the dashboard; not exercised in unit tests) ---

def _hl_post(body):
    req = urllib.request.Request(
        _HL_INFO_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


_HL_PAGE_CAP = 2000  # HL returns at most 2000 fills per userFillsByTime call


def hl_fills_fetcher(wallet, start_ms, post_fn=_hl_post):
    """Live HL `userFillsByTime` for one wallet, paginated past the 2000-fill cap.

    HL caps each response at 2000 fills; a busy wallet over a multi-day range has
    more, so a single call silently truncates (undercounts volume). We page by
    advancing `startTime` to the last fill's time and dedup by `tid` to absorb the
    boundary overlap. `post_fn` is injected for testing. Terminates when a page is
    under the cap or no new fills appear (degenerate same-timestamp page)."""
    out, seen, cur = [], set(), start_ms
    while True:
        batch = post_fn({"type": "userFillsByTime", "user": wallet, "startTime": cur})
        if not batch:
            break
        new = [f for f in batch if f.get("tid") not in seen]
        for f in new:
            seen.add(f.get("tid"))
        out.extend(new)
        if len(batch) < _HL_PAGE_CAP or not new:
            break
        last = max(int(f["time"]) for f in batch)
        cur = last if last > cur else cur + 1  # advance; re-fetch boundary, dedup handles overlap
    return out
