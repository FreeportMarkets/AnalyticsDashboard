"""Tests for HL-sourced perp volume — must be 1:1 with Hyperliquid per-fill data.

Volume = sum of |size| * price over PERP fills only (open/close, long/short),
deduped by trade id, matching exactly what HL charges builder fees on.
"""
import hl_volume as hv


# --- fill_notional: one fill's notional = |sz| * px -------------------------

def test_fill_notional_basic():
    assert hv.fill_notional({"sz": "2", "px": "100"}) == 200.0


def test_fill_notional_uses_abs_size():
    # HL sometimes carries signed/negative sizes; notional is always positive
    assert hv.fill_notional({"sz": "-3", "px": "100"}) == 300.0


def test_fill_notional_fractional():
    # real HL values are decimal strings: 12.959 * 432.13 = 5599.97267
    assert round(hv.fill_notional({"sz": "12.959", "px": "432.13"}), 2) == 5599.97


# --- is_perp_fill: only the 4 perp directions count -------------------------

def test_is_perp_fill_open_long():
    assert hv.is_perp_fill({"dir": "Open Long"}) is True


def test_is_perp_fill_close_short():
    assert hv.is_perp_fill({"dir": "Close Short"}) is True


def test_is_perp_fill_excludes_spot_buy():
    assert hv.is_perp_fill({"dir": "Buy"}) is False


def test_is_perp_fill_excludes_spot_sell():
    assert hv.is_perp_fill({"dir": "Sell"}) is False


def test_is_perp_fill_excludes_settlement():
    assert hv.is_perp_fill({"dir": "Settlement"}) is False


# --- perp_volume_from_fills: sum, perp-only, dedup --------------------------

def test_volume_sums_open_and_close():
    fills = [
        {"tid": 1, "dir": "Open Long", "sz": "2", "px": "100"},   # 200
        {"tid": 2, "dir": "Close Long", "sz": "2", "px": "150"},  # 300
    ]
    assert hv.perp_volume_from_fills(fills) == 500.0


def test_volume_counts_shorts():
    fills = [
        {"tid": 1, "dir": "Open Short", "sz": "5", "px": "10"},   # 50
        {"tid": 2, "dir": "Close Short", "sz": "5", "px": "8"},   # 40
    ]
    assert hv.perp_volume_from_fills(fills) == 90.0


def test_volume_excludes_spot_and_settlement():
    fills = [
        {"tid": 1, "dir": "Open Long", "sz": "2", "px": "100"},     # 200 counts
        {"tid": 2, "dir": "Buy", "sz": "10", "px": "100"},          # spot, skip
        {"tid": 3, "dir": "Settlement", "sz": "10", "px": "100"},   # skip
    ]
    assert hv.perp_volume_from_fills(fills) == 200.0


def test_volume_dedups_by_tid():
    # paginated fetches can return the same fill twice; count once
    fills = [
        {"tid": 7, "dir": "Open Long", "sz": "1", "px": "100"},
        {"tid": 7, "dir": "Open Long", "sz": "1", "px": "100"},
    ]
    assert hv.perp_volume_from_fills(fills) == 100.0


def test_volume_empty():
    assert hv.perp_volume_from_fills([]) == 0.0


# --- compute_perp_volume: fetch per wallet, time-window, sum ----------------

def _fake_fetcher(by_wallet):
    def fetch(wallet, start_ms):
        return by_wallet.get(wallet, [])
    return fetch


def test_compute_sums_across_wallets():
    fetcher = _fake_fetcher({
        "0xA": [{"tid": 1, "dir": "Open Long", "sz": "1", "px": "100", "time": 1000}],
        "0xB": [{"tid": 2, "dir": "Open Short", "sz": "2", "px": "100", "time": 1000}],
    })
    assert hv.compute_perp_volume(["0xA", "0xB"], 0, 2000, fetcher) == 300.0


def test_compute_filters_time_window():
    fetcher = _fake_fetcher({
        "0xA": [
            {"tid": 1, "dir": "Open Long", "sz": "1", "px": "100", "time": 500},   # before window
            {"tid": 2, "dir": "Open Long", "sz": "1", "px": "100", "time": 1500},  # in window
            {"tid": 3, "dir": "Open Long", "sz": "1", "px": "100", "time": 2500},  # after window
        ],
    })
    # window [1000, 2000)
    assert hv.compute_perp_volume(["0xA"], 1000, 2000, fetcher) == 100.0


def test_compute_dedups_same_fill_across_wallet_fetches():
    # same tid returned for two wallets (shouldn't happen, but be safe) → once
    fetcher = _fake_fetcher({
        "0xA": [{"tid": 9, "dir": "Open Long", "sz": "1", "px": "100", "time": 1000}],
        "0xB": [{"tid": 9, "dir": "Open Long", "sz": "1", "px": "100", "time": 1000}],
    })
    assert hv.compute_perp_volume(["0xA", "0xB"], 0, 2000, fetcher) == 100.0


# --- pagination: HL caps userFillsByTime at 2000 fills, must page past it -----

def test_fetcher_no_pagination_for_short_batch():
    calls = []
    def post_fn(body):
        calls.append(body["startTime"])
        return [{"tid": 1, "dir": "Open Long", "sz": "1", "px": "1", "time": 1000}]
    got = hv.hl_fills_fetcher("0xA", 0, post_fn=post_fn)
    assert len(got) == 1
    assert len(calls) == 1  # one page only


def test_fetcher_pages_past_2000_cap():
    # 2500 fills with unique increasing times; post_fn returns <=2000 from startTime
    all_fills = [{"tid": i, "dir": "Open Long", "sz": "1", "px": "1", "time": 1000 + i}
                 for i in range(2500)]
    def post_fn(body):
        s = body["startTime"]
        return [f for f in all_fills if f["time"] >= s][:2000]
    got = hv.hl_fills_fetcher("0xA", 0, post_fn=post_fn)
    assert len({f["tid"] for f in got}) == 2500  # all pages collected, deduped


def test_fetcher_terminates_on_same_timestamp_boundary():
    # 2000 fills all sharing one timestamp (degenerate) — must not infinite-loop
    all_fills = [{"tid": i, "dir": "Open Long", "sz": "1", "px": "1", "time": 5000}
                 for i in range(2000)]
    def post_fn(body):
        return all_fills  # always returns the same full page
    got = hv.hl_fills_fetcher("0xA", 0, post_fn=post_fn)
    assert len({f["tid"] for f in got}) == 2000  # deduped, loop broke
