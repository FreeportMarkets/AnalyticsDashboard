import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import requests as _requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Freeport Analytics", page_icon="📊", layout="wide")

# --- Brand Colors ---
BRAND = "#4F46E5"  # indigo
BRAND_LIGHT = "#818CF8"
ACCENT = "#10B981"  # emerald
ACCENT_WARN = "#F59E0B"  # amber
ACCENT_RED = "#EF4444"
BG_CARD = "#1E1E2E"
TEXT_MUTED = "#9CA3AF"

CHART_COLORS = [BRAND, ACCENT, ACCENT_WARN, "#EC4899", "#8B5CF6", "#06B6D4", "#F97316", "#84CC16", "#E11D48", "#14B8A6"]

# --- Custom CSS ---
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Headers */
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: rgba(79, 70, 229, 0.2);
        margin: 2rem 0;
    }

    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.2);
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB", size=13),
    margin=dict(l=0, r=0, t=40, b=0),
    hoverlabel=dict(bgcolor="#1E1E2E", font_size=13, font_color="#E5E7EB"),
)
# Default axis styling — applied separately to avoid duplicate kwarg conflicts
AXIS_DEFAULTS = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")


# --- Helpers ---
def short_wallet(w):
    if not w or len(w) < 8:
        return w or "?"
    return f"{w[:4]}...{w[-4:]}"


def fmt_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    return obj


def fmt_hour(h):
    if h == 0: return "12 AM"
    if h < 12: return f"{h} AM"
    if h == 12: return "12 PM"
    return f"{h - 12} PM"


def _detect_ostium(frame, mask):
    """Detect Ostium rows by checking category, source, or venue fields."""
    ostium = pd.Series(False, index=frame.index)
    for col in ("category", "source", "venue"):
        if col in frame.columns:
            ostium = ostium | (frame[col].str.lower() == "ostium")
    return mask & ostium


def apply_perps_leverage(frame):
    """Adjust perps volume: opens = amount*leverage, closes = size*price (position notional)."""
    if "type" not in frame.columns or "amount_usd" not in frame.columns:
        return frame
    perps_mask = frame["type"] == "perps"
    if not perps_mask.any():
        return frame
    # Determine which perps are closes
    if "is_close" in frame.columns:
        close_mask = perps_mask & frame["is_close"].astype(bool)
    else:
        close_mask = pd.Series(False, index=frame.index)
    open_mask = perps_mask & ~close_mask
    # Detect Ostium rows — amount_usd is already notional, size is already USD
    ostium_mask = _detect_ostium(frame, perps_mask)
    # Opens: collateral × leverage = notional (except Ostium where amount_usd is already notional)
    if "leverage" in frame.columns:
        frame["leverage"] = pd.to_numeric(frame["leverage"], errors="coerce").fillna(1)
        non_ostium_open = open_mask & ~ostium_mask
        frame.loc[non_ostium_open, "amount_usd"] = frame.loc[non_ostium_open, "amount_usd"] * frame.loc[non_ostium_open, "leverage"]
    # Closes: size × price = full position notional (except Ostium where size IS USD notional)
    if "size" in frame.columns and "price" in frame.columns:
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce").fillna(0)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0)
        non_ostium_close = close_mask & ~ostium_mask
        ostium_close = close_mask & ostium_mask
        frame.loc[non_ostium_close, "amount_usd"] = frame.loc[non_ostium_close, "size"].abs() * frame.loc[non_ostium_close, "price"]
        frame.loc[ostium_close, "amount_usd"] = frame.loc[ostium_close, "size"].abs()
    # Closes represent a round-trip (open + close) — store multiplier for volume calcs
    frame["_volume_usd"] = frame["amount_usd"].copy()
    frame.loc[close_mask, "_volume_usd"] = frame.loc[close_mask, "_volume_usd"] * 2
    return frame


def make_h_bar(data, x_col, y_col, title="", color=BRAND, x_prefix="", x_suffix=""):
    fig = px.bar(
        data, x=x_col, y=y_col, orientation="h",
        text=[f"{x_prefix}{fmt_number(v)}{x_suffix}" for v in data[x_col]],
    )
    fig.update_traces(
        marker_color=color, textposition="auto",
        textfont=dict(size=12, color="white"),
        hovertemplate=f"<b>%{{y}}</b><br>{x_prefix}%{{x:,.0f}}{x_suffix}<extra></extra>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, title=title, showlegend=False,
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="", **AXIS_DEFAULTS),
        height=max(300, len(data) * 32 + 60),
    )
    fig.update_yaxes(title="")
    return fig


def make_time_series(data, x_col, y_col, title="", color=BRAND, kind="area"):
    if kind == "area":
        fig = px.area(data, x=x_col, y=y_col)
        fig.update_traces(
            line=dict(color=color, width=2.5),
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        )
    else:
        fig = px.bar(data, x=x_col, y=y_col)
        fig.update_traces(marker_color=color)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, showlegend=False, height=350)
    fig.update_xaxes(title="", tickformat="%b %d", **AXIS_DEFAULTS)
    fig.update_yaxes(title="", **AXIS_DEFAULTS)
    return fig


def make_funnel(steps, values, title=""):
    fig = go.Figure(go.Funnel(
        y=steps, x=values,
        textinfo="value+percent previous",
        textfont=dict(size=14, color="white"),
        marker=dict(color=[BRAND, BRAND_LIGHT, ACCENT, ACCENT_WARN, ACCENT_RED][:len(steps)]),
        connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=350)
    return fig


# --- AWS Setup ---
@st.cache_resource
@st.cache_resource
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=st.secrets["aws"]["region"],
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    )


@st.cache_resource
def get_cloudwatch():
    return boto3.client(
        "cloudwatch",
        region_name=st.secrets["aws"]["region"],
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    )


CW_NAMESPACE = "Freeport/TradingBackend"
CW_ENVIRONMENT = "development"

RENDER_CW_NAMESPACE = "Render"
RENDER_SERVICES = ["Twitter_scraper", "Recommender", "Swap_Server-1", "TradeNews_serv", "PaymentsServer"]
CF_NAMESPACE = "AWS/CloudFront"


@st.cache_data(ttl=300, max_entries=30)
def load_cw_metric(metric_name, stat="Average", period=300, hours=24, dimensions=None):
    """Fetch a single CloudWatch metric time series."""
    try:
        cw = get_cloudwatch()
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        dims = [{"Name": "Environment", "Value": CW_ENVIRONMENT}]
        if dimensions:
            for k, v in dimensions.items():
                dims.append({"Name": k, "Value": v})
        resp = cw.get_metric_statistics(
            Namespace=CW_NAMESPACE,
            MetricName=metric_name,
            Dimensions=dims,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[stat],
        )
        points = resp.get("Datapoints", [])
        if not points:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["Timestamp"])
        df["value"] = df[stat]
        return df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "value"])


@st.cache_data(ttl=300, max_entries=10)
def load_cw_metric_by_dims(metric_name, dim_name, dim_values, stat="Sum", period=300, hours=24):
    """Fetch a CloudWatch metric broken down by a dimension."""
    frames = []
    for val in dim_values:
        df = load_cw_metric(metric_name, stat=stat, period=period, hours=hours, dimensions={dim_name: val})
        if not df.empty:
            df["dimension"] = val
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "value", "dimension"])


@st.cache_data(ttl=300, max_entries=30)
def load_render_metric(metric_name, stat="Average", period=300, hours=24, dimensions=None):
    """Fetch a metric from the Render namespace (OTel collector → EMF → CloudWatch)."""
    try:
        cw = get_cloudwatch()
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        dims = []
        if dimensions:
            for k, v in dimensions.items():
                dims.append({"Name": k, "Value": v})
        resp = cw.get_metric_statistics(
            Namespace=RENDER_CW_NAMESPACE,
            MetricName=metric_name,
            Dimensions=dims,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[stat],
        )
        points = resp.get("Datapoints", [])
        if not points:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["Timestamp"])
        df["value"] = df[stat]
        return df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "value"])


@st.cache_data(ttl=300, max_entries=10)
def load_render_metric_by_service(metric_name, stat="Average", period=300, hours=24):
    """Fetch a Render metric broken down by service.name dimension."""
    # Discover which services report this metric
    try:
        cw = get_cloudwatch()
        listed = cw.list_metrics(Namespace=RENDER_CW_NAMESPACE, MetricName=metric_name)
        svc_names = sorted(set(
            d["Value"] for m in listed.get("Metrics", [])
            for d in m.get("Dimensions", []) if d["Name"] == "service.name"
        ))
    except Exception:
        svc_names = RENDER_SERVICES
    if not svc_names:
        svc_names = RENDER_SERVICES

    frames = []
    for svc in svc_names:
        df = load_render_metric(metric_name, stat=stat, period=period, hours=hours,
                                dimensions={"service.name": svc})
        if not df.empty:
            df["service"] = svc
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "value", "service"])


@st.cache_resource
def get_logs_client():
    return boto3.client(
        "logs",
        region_name=st.secrets["aws"]["region"],
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    )


@st.cache_data(ttl=120, max_entries=5)
def query_render_logs(log_group, query, hours=1, limit=50):
    """Run a CloudWatch Insights query against a Render log group."""
    try:
        logs = get_logs_client()
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        resp = logs.start_query(
            logGroupName=log_group,
            startTime=int(start.timestamp()),
            endTime=int(end.timestamp()),
            queryString=query,
            limit=limit,
        )
        query_id = resp["queryId"]
        # Poll for results (max 10s)
        import time
        for _ in range(20):
            result = logs.get_query_results(queryId=query_id)
            if result["status"] == "Complete":
                rows = []
                for r in result["results"]:
                    row = {f["field"]: f["value"] for f in r}
                    rows.append(row)
                return pd.DataFrame(rows) if rows else pd.DataFrame()
            time.sleep(0.5)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, max_entries=10)
def load_cloudfront_metric(metric_name, stat="Sum", period=300, hours=24, dist_id=None):
    """Fetch a CloudFront metric. Discovers distribution ID if not provided."""
    try:
        cw = get_cloudwatch()
        if not dist_id:
            # Discover distribution IDs
            listed = cw.list_metrics(Namespace=CF_NAMESPACE, MetricName="Requests")
            dist_ids = sorted(set(
                d["Value"] for m in listed.get("Metrics", [])
                for d in m.get("Dimensions", []) if d["Name"] == "DistributionId"
            ))
            if not dist_ids:
                return pd.DataFrame(columns=["timestamp", "value"])
            dist_id = dist_ids[0]  # Use first distribution
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        dims = [
            {"Name": "DistributionId", "Value": dist_id},
            {"Name": "Region", "Value": "Global"},
        ]
        resp = cw.get_metric_statistics(
            Namespace=CF_NAMESPACE,
            MetricName=metric_name,
            Dimensions=dims,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[stat],
        )
        points = resp.get("Datapoints", [])
        if not points:
            return pd.DataFrame(columns=["timestamp", "value"])
        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["Timestamp"])
        df["value"] = df[stat]
        return df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "value"])


@st.cache_data(ttl=300, max_entries=1)
def get_cloudfront_dist_id():
    """Discover the CloudFront distribution ID from CloudWatch."""
    try:
        cw = get_cloudwatch()
        listed = cw.list_metrics(Namespace=CF_NAMESPACE, MetricName="Requests")
        dist_ids = sorted(set(
            d["Value"] for m in listed.get("Metrics", [])
            for d in m.get("Dimensions", []) if d["Name"] == "DistributionId"
        ))
        return dist_ids[0] if dist_ids else None
    except Exception:
        return None


@st.cache_data(ttl=120, max_entries=5)
def query_render_logs_stats(query, hours=1):
    """Run a CloudWatch Insights aggregation query, return DataFrame of results."""
    try:
        logs = get_logs_client()
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        resp = logs.start_query(
            logGroupName="/render/logs",
            startTime=int(start.timestamp()),
            endTime=int(end.timestamp()),
            queryString=query,
            limit=1000,
        )
        query_id = resp["queryId"]
        import time
        for _ in range(20):
            result = logs.get_query_results(queryId=query_id)
            if result["status"] == "Complete":
                rows = []
                for r in result["results"]:
                    row = {f["field"]: f["value"] for f in r}
                    rows.append(row)
                return pd.DataFrame(rows) if rows else pd.DataFrame()
            time.sleep(0.5)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


ANALYTICS_TABLE = "freeport-analytics-events"
TRADES_TABLE = "freeport-trades-history"


# --- Data Loading ---
@st.cache_data(ttl=300, max_entries=32)
def load_events_for_date(date_str: str) -> list:
    db = get_dynamodb()
    table = db.Table(ANALYTICS_TABLE)
    items, params = [], {"KeyConditionExpression": Key("date").eq(date_str)}
    while True:
        resp = table.query(**params)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        params["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return [decimal_to_float(i) for i in items]


@st.cache_data(ttl=300, max_entries=2)
def load_events_range(start_date: str, end_date: str) -> list:
    current = datetime.strptime(start_date, "%Y-%m-%d")
    # Fetch one extra UTC day so late-EST events (stored under next UTC date) are included
    end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    dates = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    # Fetch all days in parallel
    all_items = []
    with ThreadPoolExecutor(max_workers=min(len(dates), 8)) as pool:
        futures = {pool.submit(load_events_for_date, d): d for d in dates}
        for f in as_completed(futures):
            all_items.extend(f.result())
    return all_items


@st.cache_data(ttl=300, max_entries=32)
def load_trades_for_date(date_str: str) -> list:
    """Query trades for a single date using the trade_date GSI (no table scan)."""
    db = get_dynamodb()
    table = db.Table(TRADES_TABLE)
    items, params = [], {
        "IndexName": "trade_date-timestamp-index",
        "KeyConditionExpression": Key("trade_date").eq(date_str),
    }
    while True:
        resp = table.query(**params)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        params["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return [decimal_to_float(i) for i in items]


@st.cache_data(ttl=300, max_entries=2)
def load_trades_range(start_date: str, end_date: str) -> list:
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    all_items = []
    with ThreadPoolExecutor(max_workers=min(len(dates), 8)) as pool:
        futures = {pool.submit(load_trades_for_date, d): d for d in dates}
        for f in as_completed(futures):
            all_items.extend(f.result())
    return all_items


EVM_FEE_PAYER = "0xe39244f14AFB106255754538Cc718cAdFC0A9905"
SOL_FEE_PAYER = "DWgK5KazKbiSPKR75zm8CaoR3KebZHnSEmJxWjVvnNkr"


ARB_RPC_URLS = [
    "https://arbitrum-one-rpc.publicnode.com",
    "https://arbitrum.meowrpc.com",
    "https://1rpc.io/arb",
    "https://arb1.arbitrum.io/rpc",
]


@st.cache_data(ttl=120)
def fetch_evm_balance(address: str) -> float | None:
    """Fetch ETH balance on Arbitrum via public RPC (with fallbacks)."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "eth_getBalance",
        "params": [address, "latest"],
    }
    for rpc_url in ARB_RPC_URLS:
        try:
            resp = _requests.post(rpc_url, json=payload, timeout=10)
            result = resp.json()
            if "result" in result:
                return int(result["result"], 16) / 1e18
        except Exception:
            continue
    return None


@st.cache_data(ttl=120)
def fetch_sol_balance(address: str) -> float | None:
    """Fetch SOL balance via Solana public RPC."""
    try:
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "getBalance",
            "params": [address],
        }).encode()
        req = urllib.request.Request(
            "https://api.mainnet-beta.solana.com",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return result["result"]["value"] / 1e9
    except Exception:
        return None



def events_to_df(events: list) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df["ts"] = ts.dt.tz_convert("America/New_York")
        # Re-derive date, hour, and day_of_week in EST
        df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
        df["hour"] = df["ts"].dt.hour
        # 0=Sunday convention: pandas dayofweek 0=Mon..6=Sun → shift to 0=Sun
        df["day_of_week"] = (df["ts"].dt.dayofweek + 1) % 7
    else:
        if "hour" in df.columns:
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        if "day_of_week" in df.columns:
            df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")
    return df


def extract_session_durations(df: pd.DataFrame) -> pd.DataFrame:
    ends = df[df["event"] == "session_end"].copy()
    if ends.empty or "metadata" not in ends.columns:
        return pd.DataFrame(columns=["wallet_address", "session_id", "duration_ms", "date"])
    rows = []
    for _, r in ends.iterrows():
        meta = r.get("metadata")
        if isinstance(meta, dict) and "duration_ms" in meta:
            d = meta["duration_ms"]
            if isinstance(d, (int, float)) and d > 0:
                rows.append({
                    "wallet_address": r["wallet_address"],
                    "session_id": r.get("session_id", ""),
                    "duration_ms": d,
                    "date": r.get("date", ""),
                })
    return pd.DataFrame(rows)


# --- Sidebar ---
st.sidebar.markdown("## Freeport Analytics")
st.sidebar.markdown("---")

today = datetime.now(tz=ZoneInfo("America/New_York")).date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(today - timedelta(days=7), today),
    max_value=today,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = today

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")
num_days = (end_date - start_date).days + 1

with st.spinner("Loading data..."):
    with ThreadPoolExecutor(max_workers=2) as pool:
        events_future = pool.submit(load_events_range, start_str, end_str)
        trades_future = pool.submit(load_trades_range, start_str, end_str)
        events = events_future.result()
        trades_raw = trades_future.result()
    df = events_to_df(events)
    del events  # free raw list — df holds all we need

# Build trades DataFrame once (reused by Overview, Top Users, and Trades tabs)
if trades_raw:
    trades_df = pd.DataFrame(trades_raw)
    if "amount_usd" in trades_df.columns:
        trades_df["amount_usd"] = pd.to_numeric(trades_df["amount_usd"], errors="coerce")
    if "timestamp" in trades_df.columns:
        _ts = pd.to_datetime(trades_df["timestamp"], errors="coerce")
        if _ts.dt.tz is None:
            _ts = _ts.dt.tz_localize("UTC")
        trades_df["ts"] = _ts.dt.tz_convert("America/New_York")
        trades_df["trade_date"] = trades_df["ts"].dt.strftime("%Y-%m-%d")
        trades_df["hour"] = trades_df["ts"].dt.hour
        trades_df["dow"] = trades_df["ts"].dt.dayofweek
    trades_df = apply_perps_leverage(trades_df)
    if "_volume_usd" not in trades_df.columns:
        trades_df["_volume_usd"] = trades_df["amount_usd"].copy()
    del trades_raw  # free raw list
else:
    trades_df = pd.DataFrame()

# Filter out server/system entries from user metrics
SYSTEM_WALLETS = {"server", "unknown", "system", ""}
if not df.empty and "wallet_address" in df.columns:
    is_real_user = ~df["wallet_address"].isin(SYSTEM_WALLETS)
    if "platform" in df.columns:
        is_real_user = is_real_user & (df["platform"] != "server")
    user_df = df[is_real_user].copy()
else:
    user_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.metric("Total Events", fmt_number(len(df)))
if not user_df.empty:
    st.sidebar.metric("Unique Users", fmt_number(user_df["wallet_address"].nunique()))
    st.sidebar.metric("Date Range", f"{num_days} days")
st.sidebar.markdown("---")
st.sidebar.caption("Data refreshes every 5 minutes")

# --- Tabs ---
tab_overview, tab_users, tab_retention, tab_funnels, tab_trades, tab_notifications, tab_backend, tab_services = st.tabs(
    ["Overview", "Top Users", "Retention", "Funnels", "Trades", "Notifications", "Backend Health", "Services Health"]
)


# =====================
# TAB 1: Overview
# =====================
with tab_overview:
    if user_df.empty:
        st.info("No analytics events found for this date range. Events will appear here once the app starts sending data.")
    else:
        daily_users = user_df.groupby("date")["wallet_address"].nunique().reset_index(name="users")
        total_unique = user_df["wallet_address"].nunique()
        sessions = user_df[user_df["event"] == "session_start"]
        sess_durations = extract_session_durations(user_df)
        avg_dur = sess_durations["duration_ms"].mean() if not sess_durations.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DAU (avg)", f"{daily_users['users'].mean():.0f}" if not daily_users.empty else "0")
        col2.metric("Unique Users", fmt_number(total_unique))
        col3.metric("Sessions", fmt_number(len(sessions)))
        col4.metric("Avg Session", f"{avg_dur/1000:.0f}s" if avg_dur else "N/A")

        # Expandable unique users list
        with st.expander(f"View all {total_unique} unique users"):
            all_user_wallets = user_df.groupby("wallet_address").agg(
                events=("event", "count"),
                days_active=("date", "nunique"),
                first_seen=("date", "min"),
                last_seen=("date", "max"),
            ).sort_values("last_seen", ascending=False).reset_index()
            all_user_wallets["wallet_short"] = all_user_wallets["wallet_address"].apply(short_wallet)
            st.dataframe(
                all_user_wallets[["wallet_short", "wallet_address", "events", "days_active", "first_seen", "last_seen"]].rename(columns={
                    "wallet_short": "Wallet", "wallet_address": "Full Address",
                    "events": "Events", "days_active": "Days Active",
                    "first_seen": "First Seen", "last_seen": "Last Seen",
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")

        # DAU over time
        fig = make_time_series(daily_users, "date", "users", title="Daily Active Users", color=BRAND)
        st.plotly_chart(fig, use_container_width=True)

        # Concurrent views over time (5-min intervals, smooth curve)
        if "ts" in user_df.columns:
            cv_df = user_df.dropna(subset=["ts"]).copy()
            if not cv_df.empty:
                cv_df["bucket"] = cv_df["ts"].dt.floor("5min")
                concurrent = cv_df.groupby("bucket")["wallet_address"].nunique().reset_index()
                concurrent.columns = ["time", "viewers"]
                concurrent = concurrent.sort_values("time")
                # Smooth with rolling average for a cleaner curve
                concurrent["viewers_smooth"] = concurrent["viewers"].rolling(window=3, center=True, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=concurrent["time"], y=concurrent["viewers_smooth"],
                    mode="lines",
                    line=dict(color=BRAND, width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(79, 70, 229, 0.15)",
                    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>%{y:.0f} viewers<extra></extra>",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, title="Concurrent Views (5-min intervals, EST)", height=350)
                fig.update_xaxes(title="", tickformat="%b %d, %I %p", **AXIS_DEFAULTS)
                fig.update_yaxes(title="Active Users", **AXIS_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True)

        # WAU / MAU side by side
        col_w, col_m = st.columns(2)
        wau = user_df[user_df["date"] >= (today - timedelta(days=7)).strftime("%Y-%m-%d")]["wallet_address"].nunique()
        mau = user_df[user_df["date"] >= (today - timedelta(days=30)).strftime("%Y-%m-%d")]["wallet_address"].nunique()
        col_w.metric("WAU (7d)", fmt_number(wau))
        col_m.metric("MAU (30d)", fmt_number(mau))

        st.markdown("---")

        # --- Trading Snapshot + User Insights ---
        if not trades_df.empty:
            trades_only_ov = trades_df[trades_df["type"].isin(["swap", "perps"])] if "type" in trades_df.columns else trades_df
            deposits_ov = trades_df[trades_df["type"] == "deposit"] if "type" in trades_df.columns else pd.DataFrame()

            trade_vol = trades_only_ov["_volume_usd"].sum() if not trades_only_ov.empty and "_volume_usd" in trades_only_ov.columns else 0
            dep_vol = deposits_ov["_volume_usd"].sum() if not deposits_ov.empty and "_volume_usd" in deposits_ov.columns else 0
            num_traders = trades_only_ov["wallet_address"].nunique() if not trades_only_ov.empty and "wallet_address" in trades_only_ov.columns else 0
            trader_pct = (num_traders / total_unique * 100) if total_unique > 0 else 0

            tv1, tv2, tv3, tv4 = st.columns(4)
            tv1.metric("Trading Volume", f"${fmt_number(trade_vol)}")
            tv2.metric("Deposit Volume", f"${fmt_number(dep_vol)}")
            tv3.metric("Active Traders", fmt_number(num_traders))
            tv4.metric("Trader %", f"{trader_pct:.0f}%")

        # New vs Returning users
        # Users whose first-ever event is within this date range = new
        first_seen_dates = user_df.groupby("wallet_address")["date"].min()
        new_users = first_seen_dates[first_seen_dates >= start_str].count()
        returning_users = total_unique - new_users

        # Events per user
        avg_events_per_user = len(user_df) / total_unique if total_unique > 0 else 0

        nu1, nu2, nu3 = st.columns(3)
        nu1.metric("New Users", fmt_number(new_users), delta=f"{new_users / total_unique * 100:.0f}% of total" if total_unique > 0 else None)
        nu2.metric("Returning Users", fmt_number(returning_users))
        nu3.metric("Events / User", f"{avg_events_per_user:.1f}")

        # --- iOS vs Android Comparison ---
        if "platform" in user_df.columns:
            platforms = user_df["platform"].str.lower().unique()
            has_ios = "ios" in platforms
            has_android = "android" in platforms

            if has_ios or has_android:
                st.subheader("iOS vs Android")

                ios_df = user_df[user_df["platform"].str.lower() == "ios"]
                android_df = user_df[user_df["platform"].str.lower() == "android"]

                ios_users = ios_df["wallet_address"].nunique()
                android_users = android_df["wallet_address"].nunique()

                ios_sessions = extract_session_durations(ios_df)
                android_sessions = extract_session_durations(android_df)
                ios_avg_sess = ios_sessions["duration_ms"].mean() / 1000 if not ios_sessions.empty else 0
                android_avg_sess = android_sessions["duration_ms"].mean() / 1000 if not android_sessions.empty else 0

                ios_events_per = len(ios_df) / ios_users if ios_users > 0 else 0
                android_events_per = len(android_df) / android_users if android_users > 0 else 0

                # Metric comparison row
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("iOS Users", fmt_number(ios_users))
                mc2.metric("Android Users", fmt_number(android_users))
                mc3.metric("iOS Avg Session", f"{ios_avg_sess:.0f}s")
                mc4.metric("Android Avg Session", f"{android_avg_sess:.0f}s")

                mc5, mc6, mc7, mc8 = st.columns(4)
                mc5.metric("iOS Events/User", f"{ios_events_per:.1f}")
                mc6.metric("Android Events/User", f"{android_events_per:.1f}")
                mc7.metric("iOS Sessions", fmt_number(len(ios_sessions)))
                mc8.metric("Android Sessions", fmt_number(len(android_sessions)))

                # DAU by platform over time
                if not ios_df.empty or not android_df.empty:
                    ios_dau = ios_df.groupby("date")["wallet_address"].nunique().reset_index(name="iOS")
                    android_dau = android_df.groupby("date")["wallet_address"].nunique().reset_index(name="Android")
                    dau_merged = pd.merge(ios_dau, android_dau, on="date", how="outer").fillna(0).sort_values("date")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dau_merged["date"], y=dau_merged["iOS"], name="iOS",
                                            mode="lines+markers", line=dict(color=BRAND, width=2.5),
                                            marker=dict(size=6)))
                    fig.add_trace(go.Scatter(x=dau_merged["date"], y=dau_merged["Android"], name="Android",
                                            mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                                            marker=dict(size=6)))
                    fig.update_layout(**PLOTLY_LAYOUT, title="Daily Active Users by Platform", height=350,
                                      legend=dict(orientation="h", y=-0.15))
                    fig.update_xaxes(title="", tickformat="%b %d", **AXIS_DEFAULTS)
                    fig.update_yaxes(title="Users", **AXIS_DEFAULTS)
                    st.plotly_chart(fig, use_container_width=True)

                # Top events comparison side by side
                col_ios_ev, col_and_ev = st.columns(2)
                with col_ios_ev:
                    if not ios_df.empty:
                        ios_top = ios_df["event"].value_counts().head(10).reset_index()
                        ios_top.columns = ["event", "count"]
                        fig = make_h_bar(ios_top, "count", "event", title="Top iOS Events", color=BRAND)
                        st.plotly_chart(fig, use_container_width=True)
                with col_and_ev:
                    if not android_df.empty:
                        and_top = android_df["event"].value_counts().head(10).reset_index()
                        and_top.columns = ["event", "count"]
                        fig = make_h_bar(and_top, "count", "event", title="Top Android Events", color=ACCENT)
                        st.plotly_chart(fig, use_container_width=True)

                # Trading comparison
                if not trades_df.empty and "wallet_address" in trades_df.columns:
                    ios_wallets = set(ios_df["wallet_address"].unique())
                    android_wallets = set(android_df["wallet_address"].unique())
                    trade_only = trades_df[trades_df["type"].isin(["swap", "perps"])] if "type" in trades_df.columns else trades_df

                    ios_trades = trade_only[trade_only["wallet_address"].isin(ios_wallets)]
                    android_trades = trade_only[trade_only["wallet_address"].isin(android_wallets)]

                    ios_vol = ios_trades["_volume_usd"].sum() if not ios_trades.empty and "_volume_usd" in ios_trades.columns else 0
                    android_vol = android_trades["_volume_usd"].sum() if not android_trades.empty and "_volume_usd" in android_trades.columns else 0
                    ios_trade_ct = len(ios_trades)
                    android_trade_ct = len(android_trades)

                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("iOS Volume", f"${fmt_number(ios_vol)}")
                    tc2.metric("Android Volume", f"${fmt_number(android_vol)}")
                    tc3.metric("iOS Trades", fmt_number(ios_trade_ct))
                    tc4.metric("Android Trades", fmt_number(android_trade_ct))

                    # Perps breakdown by platform
                    if "type" in trades_df.columns:
                        perps_all = trades_df[trades_df["type"] == "perps"]
                        if not perps_all.empty:
                            ios_perps = perps_all[perps_all["wallet_address"].isin(ios_wallets)]
                            android_perps = perps_all[perps_all["wallet_address"].isin(android_wallets)]

                            ios_p_vol = ios_perps["_volume_usd"].sum() if not ios_perps.empty else 0
                            android_p_vol = android_perps["_volume_usd"].sum() if not android_perps.empty else 0
                            ios_p_ct = len(ios_perps)
                            android_p_ct = len(android_perps)
                            ios_p_traders = ios_perps["wallet_address"].nunique() if not ios_perps.empty else 0
                            android_p_traders = android_perps["wallet_address"].nunique() if not android_perps.empty else 0
                            ios_p_avg = ios_p_vol / ios_p_ct if ios_p_ct > 0 else 0
                            android_p_avg = android_p_vol / android_p_ct if android_p_ct > 0 else 0

                            st.markdown("**Perpetuals**")
                            pp1, pp2, pp3, pp4 = st.columns(4)
                            pp1.metric("iOS Perps Volume", f"${fmt_number(ios_p_vol)}")
                            pp2.metric("Android Perps Volume", f"${fmt_number(android_p_vol)}")
                            pp3.metric("iOS Perps Traders", fmt_number(ios_p_traders))
                            pp4.metric("Android Perps Traders", fmt_number(android_p_traders))

                            pp5, pp6, pp7, pp8 = st.columns(4)
                            pp5.metric("iOS Orders", fmt_number(ios_p_ct))
                            pp6.metric("Android Orders", fmt_number(android_p_ct))
                            pp7.metric("iOS Avg Order", f"${ios_p_avg:,.0f}")
                            pp8.metric("Android Avg Order", f"${android_p_avg:,.0f}")

                            # Leverage comparison
                            if "leverage" in perps_all.columns:
                                ios_lev = pd.to_numeric(ios_perps["leverage"], errors="coerce").dropna() if not ios_perps.empty else pd.Series(dtype=float)
                                and_lev = pd.to_numeric(android_perps["leverage"], errors="coerce").dropna() if not android_perps.empty else pd.Series(dtype=float)
                                lv1, lv2 = st.columns(2)
                                lv1.metric("iOS Avg Leverage", f"{ios_lev.mean():.1f}x" if not ios_lev.empty else "N/A")
                                lv2.metric("Android Avg Leverage", f"{and_lev.mean():.1f}x" if not and_lev.empty else "N/A")

                            # Long/Short split per platform
                            if "side" in perps_all.columns:
                                col_ios_ls, col_and_ls = st.columns(2)
                                with col_ios_ls:
                                    if not ios_perps.empty:
                                        ios_sides = ios_perps["side"].value_counts().reset_index()
                                        ios_sides.columns = ["side", "count"]
                                        colors = {"Long": ACCENT, "Short": ACCENT_RED, "long": ACCENT, "short": ACCENT_RED}
                                        fig = px.pie(ios_sides, values="count", names="side", hole=0.5,
                                                     color="side", color_discrete_map=colors)
                                        fig.update_traces(textinfo="label+percent+value", textfont=dict(size=13, color="white"))
                                        fig.update_layout(**PLOTLY_LAYOUT, title="iOS Long vs Short", height=300, showlegend=False)
                                        st.plotly_chart(fig, use_container_width=True)
                                with col_and_ls:
                                    if not android_perps.empty:
                                        and_sides = android_perps["side"].value_counts().reset_index()
                                        and_sides.columns = ["side", "count"]
                                        fig = px.pie(and_sides, values="count", names="side", hole=0.5,
                                                     color="side", color_discrete_map=colors)
                                        fig.update_traces(textinfo="label+percent+value", textfont=dict(size=13, color="white"))
                                        fig.update_layout(**PLOTLY_LAYOUT, title="Android Long vs Short", height=300, showlegend=False)
                                        st.plotly_chart(fig, use_container_width=True)

                            # Top assets per platform
                            if "asset" in perps_all.columns:
                                col_ios_a, col_and_a = st.columns(2)
                                with col_ios_a:
                                    if not ios_perps.empty and "_volume_usd" in ios_perps.columns:
                                        ios_assets = ios_perps.groupby("asset")["_volume_usd"].sum().sort_values(ascending=False).head(10).reset_index()
                                        ios_assets.columns = ["asset", "volume"]
                                        ios_assets["volume"] = ios_assets["volume"].round(0)
                                        fig = make_h_bar(ios_assets, "volume", "asset", title="iOS Top Assets", color=BRAND, x_prefix="$")
                                        st.plotly_chart(fig, use_container_width=True)
                                with col_and_a:
                                    if not android_perps.empty and "_volume_usd" in android_perps.columns:
                                        and_assets = android_perps.groupby("asset")["_volume_usd"].sum().sort_values(ascending=False).head(10).reset_index()
                                        and_assets.columns = ["asset", "volume"]
                                        and_assets["volume"] = and_assets["volume"].round(0)
                                        fig = make_h_bar(and_assets, "volume", "asset", title="Android Top Assets", color=ACCENT, x_prefix="$")
                                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Events by day + Top events side by side
        col_left, col_right = st.columns([3, 2])

        with col_left:
            ebd = user_df.groupby("date").size().reset_index(name="count")
            fig = make_time_series(ebd, "date", "count", title="Events by Day", color=BRAND_LIGHT, kind="bar")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            top_ev = user_df["event"].value_counts().head(12).reset_index()
            top_ev.columns = ["event", "count"]
            fig = make_h_bar(top_ev, "count", "event", title="Top Events", color=BRAND)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Hourly + DOW side by side
        col_h, col_d = st.columns(2)

        with col_h:
            if "hour" in user_df.columns:
                hourly = user_df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
                hourly.columns = ["hour", "events"]
                hourly["label"] = hourly["hour"].apply(fmt_hour)
                fig = px.bar(hourly, x="label", y="events", title="Hourly Activity (EST)")
                fig.update_traces(marker_color=BRAND, hovertemplate="<b>%{x}</b><br>%{y:,} events<extra></extra>")
                fig.update_layout(**PLOTLY_LAYOUT, height=320)
                fig.update_xaxes(title="", tickangle=-45)
                fig.update_yaxes(title="")
                st.plotly_chart(fig, use_container_width=True)

        with col_d:
            if "day_of_week" in user_df.columns:
                dow_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                by_dow = user_df.groupby("day_of_week").size().reindex(range(7), fill_value=0).reset_index()
                by_dow.columns = ["dow", "events"]
                by_dow["day"] = by_dow["dow"].map(lambda i: dow_labels[int(i)] if pd.notna(i) and int(i) < 7 else "?")
                fig = px.bar(by_dow, x="day", y="events", title="Activity by Day of Week")
                fig.update_traces(marker_color=ACCENT, hovertemplate="<b>%{x}</b><br>%{y:,} events<extra></extra>")
                fig.update_layout(**PLOTLY_LAYOUT, height=320)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="")
                st.plotly_chart(fig, use_container_width=True)


# =====================
# TAB 2: Top Users
# =====================
with tab_users:
    if user_df.empty:
        st.info("No data available.")
    else:
        # --- Time Spent Leaderboard ---
        st.subheader("Time Spent in App")
        sess_durations = extract_session_durations(user_df)
        if not sess_durations.empty:
            time_by_user = sess_durations.groupby("wallet_address").agg(
                total_time_min=("duration_ms", lambda x: round(x.sum() / 60000, 1)),
                sessions=("session_id", "count"),
                avg_session_min=("duration_ms", lambda x: round(x.mean() / 60000, 1)),
            ).sort_values("total_time_min", ascending=False).head(20).reset_index()
            time_by_user["wallet"] = time_by_user["wallet_address"].apply(short_wallet)

            fig = make_h_bar(
                time_by_user, "total_time_min", "wallet",
                title="Top Users by Time Spent", color=BRAND, x_suffix=" min",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detailed time breakdown"):
                st.dataframe(
                    time_by_user[["wallet", "total_time_min", "sessions", "avg_session_min"]].rename(columns={
                        "wallet": "Wallet", "total_time_min": "Total (min)",
                        "sessions": "Sessions", "avg_session_min": "Avg (min)",
                    }),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No session duration data yet.")

        st.markdown("---")

        # --- Most Active by Event Count ---
        st.subheader("Most Active Users")
        events_per_user = user_df.groupby("wallet_address").agg(
            events=("event", "count"),
            days_active=("date", "nunique"),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
        ).sort_values("events", ascending=False).head(20).reset_index()
        events_per_user["wallet"] = events_per_user["wallet_address"].apply(short_wallet)

        fig = make_h_bar(events_per_user, "events", "wallet", title="Top Users by Event Count", color=ACCENT)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed activity breakdown"):
            st.dataframe(
                events_per_user[["wallet", "events", "days_active", "first_seen", "last_seen"]].rename(columns={
                    "wallet": "Wallet", "events": "Events", "days_active": "Days Active",
                    "first_seen": "First Seen", "last_seen": "Last Seen",
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")

        # --- Top Traders ---
        st.subheader("Top Traders")
        if not trades_df.empty:
            tdf = trades_df[trades_df["type"] != "deposit"] if "type" in trades_df.columns else trades_df
            if not tdf.empty and "wallet_address" in tdf.columns:

                col_vol, col_count = st.columns(2)

                with col_vol:
                    vol_col = "_volume_usd" if "_volume_usd" in tdf.columns else "amount_usd"
                    trader_vol = tdf.groupby("wallet_address").agg(
                        volume=(vol_col, "sum"), trades=(vol_col, "count"),
                    ).sort_values("volume", ascending=False).head(15).reset_index()
                    trader_vol["wallet"] = trader_vol["wallet_address"].apply(short_wallet)
                    trader_vol["volume"] = trader_vol["volume"].round(0)
                    fig = make_h_bar(trader_vol, "volume", "wallet", title="By Volume ($)", color=ACCENT_WARN, x_prefix="$")
                    st.plotly_chart(fig, use_container_width=True)

                with col_count:
                    trader_cnt = tdf.groupby("wallet_address").size().sort_values(ascending=False).head(15).reset_index()
                    trader_cnt.columns = ["wallet_address", "trades"]
                    trader_cnt["wallet"] = trader_cnt["wallet_address"].apply(short_wallet)
                    fig = make_h_bar(trader_cnt, "trades", "wallet", title="By Trade Count", color="#EC4899")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trade data for this period.")
        else:
            st.info("No trade data for this period.")

        st.markdown("---")

        # --- Heatmap ---
        st.subheader("Activity Heatmap")
        if "hour" in user_df.columns and "day_of_week" in user_df.columns:
            dow_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            heatmap = user_df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
            heatmap = heatmap.reindex(index=range(7), columns=range(24), fill_value=0)

            fig = go.Figure(data=go.Heatmap(
                z=heatmap.values,
                x=[fmt_hour(h) for h in range(24)],
                y=dow_labels,
                colorscale=[[0, "#1a1a2e"], [0.5, BRAND], [1, "#EC4899"]],
                hovertemplate="<b>%{y} %{x}</b><br>%{z} events<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, title="Events by Hour & Day of Week (EST)", height=300,
            )
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- Per-User Deep Dive ---
        st.subheader("User Deep Dive")
        all_wallets = sorted(user_df["wallet_address"].unique())
        wallet_options = [f"{short_wallet(w)} ({w})" for w in all_wallets]
        selected = st.selectbox("Select a user", wallet_options, index=None, placeholder="Pick a wallet...")

        if selected:
            full_wallet = selected.split("(")[1].rstrip(")")
            selected_user_df = user_df[user_df["wallet_address"] == full_wallet]

            col1, col2, col3 = st.columns(3)
            col1.metric("Events", fmt_number(len(selected_user_df)))
            col2.metric("Days Active", str(selected_user_df["date"].nunique()))
            user_sessions = extract_session_durations(selected_user_df)
            total_min = user_sessions["duration_ms"].sum() / 60000 if not user_sessions.empty else 0
            col3.metric("Total Time", f"{total_min:.1f} min")

            col_ev, col_act = st.columns(2)
            with col_ev:
                ue = selected_user_df["event"].value_counts().head(10).reset_index()
                ue.columns = ["event", "count"]
                fig = make_h_bar(ue, "count", "event", title="Event Breakdown", color=BRAND)
                st.plotly_chart(fig, use_container_width=True)

            with col_act:
                ud = selected_user_df.groupby("date").size().reset_index(name="events")
                fig = make_time_series(ud, "date", "events", title="Daily Activity", color=ACCENT, kind="bar")
                st.plotly_chart(fig, use_container_width=True)

            if not trades_df.empty and "wallet_address" in trades_df.columns:
                utdf = trades_df[trades_df["wallet_address"] == full_wallet]
                if not utdf.empty:
                    st.markdown(f"**Trades:** {len(utdf)} trades | **${utdf['_volume_usd'].sum():,.2f}** total volume")
                    display_cols = [c for c in ["timestamp", "from_token", "to_token", "amount_usd", "status", "source"] if c in utdf.columns]
                    display_df = utdf[display_cols].sort_values("timestamp", ascending=False).copy()
                    if "timestamp" in display_df.columns:
                        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%b %d, %I:%M %p")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =====================
# TAB 3: Retention
# =====================
with tab_retention:
    if user_df.empty:
        st.info("No data for retention analysis. Need at least a few days of data for cohort analysis.")
    else:
        first_seen = user_df.groupby("wallet_address")["date"].min().reset_index()
        first_seen.columns = ["wallet_address", "cohort_date"]
        cohort_dates = sorted(first_seen["cohort_date"].unique())
        retention_days = [0, 1, 3, 7, 14, 30]

        retention_data = []
        retention_nums = []
        for cohort in cohort_dates:
            cohort_users = set(first_seen[first_seen["cohort_date"] == cohort]["wallet_address"])
            cohort_size = len(cohort_users)
            if cohort_size == 0:
                continue
            row = {"Cohort": cohort, "Users": cohort_size}
            num_row = {"cohort": cohort, "size": cohort_size}
            cohort_dt = datetime.strptime(cohort, "%Y-%m-%d")
            for d in retention_days:
                target_date = (cohort_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                active_on_day = set(user_df[user_df["date"] == target_date]["wallet_address"])
                retained = cohort_users & active_on_day
                pct = (len(retained) / cohort_size * 100) if cohort_size > 0 else 0
                row[f"D{d}"] = f"{pct:.0f}%"
                num_row[f"D{d}"] = pct
            retention_data.append(row)
            retention_nums.append(num_row)

        if retention_data:
            # Styled retention table
            ret_df = pd.DataFrame(retention_data)
            st.dataframe(ret_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Retention curve
            st.subheader("Average Retention Curve")
            avg_ret = {}
            for d in retention_days:
                vals = [r[f"D{d}"] for r in retention_nums]
                avg_ret[d] = sum(vals) / len(vals) if vals else 0

            curve_df = pd.DataFrame({"Day": [f"D{d}" for d in retention_days], "Retention %": list(avg_ret.values())})
            fig = px.line(curve_df, x="Day", y="Retention %", markers=True)
            fig.update_traces(
                line=dict(color=BRAND, width=3),
                marker=dict(size=10, color=BRAND),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            fig.update_yaxes(range=[0, 105], **AXIS_DEFAULTS)
            fig.update_xaxes(title="")
            fig.update_yaxes(title="Retention %")
            st.plotly_chart(fig, use_container_width=True)

            # Retention heatmap
            if len(retention_nums) > 1:
                st.subheader("Cohort Retention Heatmap")
                heat_data = []
                for r in retention_nums:
                    for d in retention_days:
                        heat_data.append({"Cohort": r["cohort"], "Day": f"D{d}", "Retention": r[f"D{d}"]})
                heat_df = pd.DataFrame(heat_data)
                pivot = heat_df.pivot(index="Cohort", columns="Day", values="Retention")
                pivot = pivot[[f"D{d}" for d in retention_days]]

                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    colorscale=[[0, "#1a1a2e"], [0.5, BRAND], [1, ACCENT]],
                    text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
                    texttemplate="%{text}", textfont=dict(size=12, color="white"),
                    hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>%{z:.0f}% retained<extra></extra>",
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=max(250, len(pivot) * 35 + 80),
                )
                fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
                fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for cohort analysis.")


# =====================
# TAB 4: Funnels
# =====================
with tab_funnels:
    if user_df.empty:
        st.info("No data for funnel analysis.")
    else:
        col_notif, col_trade = st.columns(2)

        # Notification → Trade funnel
        with col_notif:
            st.subheader("Notification → Trade")
            notif_received = user_df[user_df["event"] == "notification_received"]["wallet_address"].nunique()
            notif_tapped = user_df[user_df["event"] == "notification_tap"]["wallet_address"].nunique()
            trade_init = user_df[user_df["event"] == "trade_initiated"]["wallet_address"].nunique()
            trade_ok = user_df[user_df["event"] == "trade_success"]["wallet_address"].nunique()
            fig = make_funnel(
                ["Notif Received", "Notif Tapped", "Trade Started", "Trade Success"],
                [notif_received, notif_tapped, trade_init, trade_ok],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Trade funnel
        with col_trade:
            st.subheader("Trade Funnel")
            buy_tap = user_df[user_df["event"] == "buy_button_tap"]["wallet_address"].nunique()
            swap_modal = user_df[user_df["event"] == "swap_modal_open"]["wallet_address"].nunique()
            t_init = user_df[user_df["event"] == "trade_initiated"]["wallet_address"].nunique()
            t_ok = user_df[user_df["event"] == "trade_success"]["wallet_address"].nunique()
            t_err = user_df[user_df["event"] == "trade_error"]["wallet_address"].nunique()
            fig = make_funnel(
                ["Buy Tap", "Swap Modal", "Trade Started", "Trade Success", "Trade Error"],
                [buy_tap, swap_modal, t_init, t_ok, t_err],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Deposit funnel
        col_dep, col_prov = st.columns(2)

        with col_dep:
            st.subheader("Deposit Funnel")
            dep_modal = user_df[user_df["event"] == "deposit_modal_open"]["wallet_address"].nunique()
            dep_init = user_df[user_df["event"] == "deposit_initiated"]["wallet_address"].nunique()
            dep_ok = user_df[user_df["event"] == "deposit_success"]["wallet_address"].nunique()
            dep_err = user_df[user_df["event"] == "deposit_error"]["wallet_address"].nunique()
            fig = make_funnel(
                ["Modal Open", "Initiated", "Success", "Error"],
                [dep_modal, dep_init, dep_ok, dep_err],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Deposit by provider
        with col_prov:
            st.subheader("Deposits by Provider")
            dep_events = user_df[user_df["event"].isin(["deposit_initiated", "deposit_success", "deposit_error"])]
            if not dep_events.empty and "metadata" in dep_events.columns:
                provider_rows = []
                for _, r in dep_events.iterrows():
                    meta = r.get("metadata")
                    if isinstance(meta, dict) and meta.get("provider"):
                        provider_rows.append({"provider": meta["provider"], "event": r["event"]})
                if provider_rows:
                    prov_df = pd.DataFrame(provider_rows)
                    fig = px.histogram(prov_df, x="provider", color="event", barmode="group",
                                       color_discrete_sequence=[BRAND, ACCENT, ACCENT_RED])
                    fig.update_layout(**PLOTLY_LAYOUT, height=350, legend=dict(orientation="h", y=-0.15))
                    fig.update_xaxes(title="")
                    fig.update_yaxes(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No provider metadata found.")
            else:
                st.info("No deposit events yet.")

        st.markdown("---")

        # Feature engagement
        st.subheader("Feature Engagement")
        engagement_events = [
            "feed_card_tap", "token_detail_view", "search_query", "category_select",
            "chart_period_change", "scroll_depth", "side_drawer_open",
            "perps_order_placed", "bridge_transfer", "app_open_deep_link",
        ]
        eng_data = []
        for evt in engagement_events:
            evt_df = user_df[user_df["event"] == evt]
            eng_data.append({"event": evt, "total": len(evt_df), "users": evt_df["wallet_address"].nunique()})
        eng_df = pd.DataFrame(eng_data).sort_values("total", ascending=False)
        eng_df = eng_df[eng_df["total"] > 0]

        if not eng_df.empty:
            fig = make_h_bar(eng_df, "total", "event", title="Feature Usage (total events)", color=BRAND_LIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature engagement data yet.")


# =====================
# TAB 5: Trades
# =====================
with tab_trades:
    if trades_df.empty:
        st.info("No trades found for this date range.")
    else:
        tdf = trades_df

        # Split by type (views, not copies)
        swap_df = tdf[tdf["type"] == "swap"] if "type" in tdf.columns else tdf
        perps_df = tdf[tdf["type"] == "perps"] if "type" in tdf.columns else pd.DataFrame()
        deposit_df = tdf[tdf["type"] == "deposit"] if "type" in tdf.columns else pd.DataFrame()

        trades_only = pd.concat([swap_df, perps_df], ignore_index=True)  # exclude deposits

        # --- Top-level metrics (swaps + perps only) ---
        total_vol = trades_only["_volume_usd"].sum() if "_volume_usd" in trades_only.columns and not trades_only.empty else 0
        total_trades = len(trades_only)
        unique_traders = trades_only["wallet_address"].nunique() if "wallet_address" in trades_only.columns and not trades_only.empty else 0
        avg_trade = total_vol / total_trades if total_trades > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trading Volume", f"${fmt_number(total_vol)}")
        col2.metric("Total Trades", fmt_number(total_trades))
        col3.metric("Unique Traders", fmt_number(unique_traders))
        col4.metric("Avg Trade Size", f"${avg_trade:,.0f}")

        # Type breakdown row
        swap_vol = swap_df["_volume_usd"].sum() if not swap_df.empty and "_volume_usd" in swap_df.columns else 0
        perps_vol = perps_df["_volume_usd"].sum() if not perps_df.empty and "_volume_usd" in perps_df.columns else 0
        deposit_vol = deposit_df["_volume_usd"].sum() if not deposit_df.empty and "_volume_usd" in deposit_df.columns else 0
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Swap Volume", f"${fmt_number(swap_vol)}", delta=f"{len(swap_df)} swaps")
        cb2.metric("Perps Volume", f"${fmt_number(perps_vol)}", delta=f"{len(perps_df)} orders")
        cb3.metric("Deposit Volume", f"${fmt_number(deposit_vol)}", delta=f"{len(deposit_df)} deposits")

        # --- Daily Trade Count (swaps + perps) ---
        if not trades_only.empty and "trade_date" in trades_only.columns:
            daily_counts = trades_only.groupby("trade_date").agg(
                trades=("trade_date", "size"),
                volume=("_volume_usd", "sum"),
            ).reset_index()
            daily_counts.columns = ["date", "trades", "volume"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_counts["date"], y=daily_counts["trades"],
                name="Trades",
                marker_color=BRAND,
                hovertemplate="<b>%{x}</b><br>%{y} trades<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=daily_counts["date"], y=daily_counts["volume"],
                name="Volume ($)",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Daily Trades & Volume",
                height=380,
                yaxis=dict(title="Trades", **AXIS_DEFAULTS),
                yaxis2=dict(title="Volume ($)", overlaying="y", side="right", **AXIS_DEFAULTS),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode="group",
            )
            fig.update_xaxes(title="", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

        # ============================
        # SWAPS SECTION
        # ============================
        st.markdown("---")
        st.subheader("Swaps")

        if swap_df.empty:
            st.info("No swap data for this period.")
        else:
            # Daily swap volume
            if "trade_date" in swap_df.columns and "amount_usd" in swap_df.columns:
                daily_vol = swap_df.groupby("trade_date")["amount_usd"].sum().reset_index()
                daily_vol.columns = ["date", "volume"]
                fig = make_time_series(daily_vol, "date", "volume", title="Daily Swap Volume ($)", color=ACCENT, kind="bar")
                st.plotly_chart(fig, use_container_width=True)

            # Volume by token + Count by token
            col_vol, col_cnt = st.columns(2)

            with col_vol:
                if "to_token" in swap_df.columns and "amount_usd" in swap_df.columns:
                    vbt = swap_df.groupby("to_token")["amount_usd"].sum().sort_values(ascending=False).head(15).reset_index()
                    vbt.columns = ["token", "volume"]
                    vbt["volume"] = vbt["volume"].round(0)
                    fig = make_h_bar(vbt, "volume", "token", title="Swap Volume by Token (Top 15)", color=ACCENT_WARN, x_prefix="$")
                    st.plotly_chart(fig, use_container_width=True)

            with col_cnt:
                if "to_token" in swap_df.columns:
                    cbt = swap_df["to_token"].value_counts().head(15).reset_index()
                    cbt.columns = ["token", "trades"]
                    fig = make_h_bar(cbt, "trades", "token", title="Swaps by Token (Top 15)", color="#EC4899")
                    st.plotly_chart(fig, use_container_width=True)

            # Hour + DOW
            col_h, col_d = st.columns(2)

            with col_h:
                if "hour" in swap_df.columns:
                    bh = swap_df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
                    bh.columns = ["hour", "trades"]
                    bh["label"] = bh["hour"].apply(fmt_hour)
                    fig = px.bar(bh, x="label", y="trades", title="Swaps by Hour (EST)")
                    fig.update_traces(marker_color=BRAND, hovertemplate="<b>%{x}</b><br>%{y} swaps<extra></extra>")
                    fig.update_layout(**PLOTLY_LAYOUT, height=320)
                    fig.update_xaxes(title="", tickangle=-45)
                    fig.update_yaxes(title="")
                    st.plotly_chart(fig, use_container_width=True)

            with col_d:
                if "dow" in swap_df.columns:
                    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    bd = swap_df.groupby("dow").size().reindex(range(7), fill_value=0).reset_index()
                    bd.columns = ["dow", "trades"]
                    bd["day"] = bd["dow"].map(lambda i: dow_labels[int(i)] if pd.notna(i) and int(i) < 7 else "?")
                    fig = px.bar(bd, x="day", y="trades", title="Swaps by Day of Week")
                    fig.update_traces(marker_color=ACCENT, hovertemplate="<b>%{x}</b><br>%{y} swaps<extra></extra>")
                    fig.update_layout(**PLOTLY_LAYOUT, height=320)
                    fig.update_xaxes(title="")
                    fig.update_yaxes(title="")
                    st.plotly_chart(fig, use_container_width=True)

            # Swap source pie
            if "source" in swap_df.columns:
                st.subheader("Swap Source")
                src = swap_df["source"].value_counts().reset_index()
                src.columns = ["source", "count"]
                fig = px.pie(src, values="count", names="source", hole=0.5, color_discrete_sequence=CHART_COLORS)
                fig.update_traces(textinfo="label+percent", textfont=dict(size=13, color="white"))
                fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Recent swaps table
            st.subheader("Recent Swaps")
            swap_cols = [c for c in ["timestamp", "wallet_address", "from_token", "to_token", "amount_usd", "status", "source"] if c in swap_df.columns]
            recent_swaps = swap_df[swap_cols].sort_values("timestamp", ascending=False).head(50).copy()
            if "wallet_address" in recent_swaps.columns:
                recent_swaps["wallet_address"] = recent_swaps["wallet_address"].apply(short_wallet)
            if "timestamp" in recent_swaps.columns:
                recent_swaps["timestamp"] = pd.to_datetime(recent_swaps["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%b %d, %I:%M %p")
            st.dataframe(recent_swaps, use_container_width=True, hide_index=True)

        # ============================
        # PERPS SECTION
        # ============================
        st.markdown("---")
        st.subheader("Perpetuals")

        if perps_df.empty:
            st.info("No perps data for this period.")
        else:
            # Perps metrics
            perps_total_vol = perps_df["_volume_usd"].sum() if "_volume_usd" in perps_df.columns else 0
            perps_total_orders = len(perps_df)
            perps_unique = perps_df["wallet_address"].nunique() if "wallet_address" in perps_df.columns else 0
            perps_avg = perps_total_vol / perps_total_orders if perps_total_orders > 0 else 0

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Perps Volume", f"${fmt_number(perps_total_vol)}")
            pc2.metric("Total Orders", fmt_number(perps_total_orders))
            pc3.metric("Unique Perps Traders", fmt_number(perps_unique))
            pc4.metric("Avg Order Size", f"${perps_avg:,.0f}")

            # Daily perps volume
            if "trade_date" in perps_df.columns and "_volume_usd" in perps_df.columns:
                daily_perps = perps_df.groupby("trade_date")["_volume_usd"].sum().reset_index()
                daily_perps.columns = ["date", "volume"]
                fig = make_time_series(daily_perps, "date", "volume", title="Daily Perps Volume ($)", color="#8B5CF6", kind="bar")
                st.plotly_chart(fig, use_container_width=True)

            # Volume by Asset + Orders by Asset
            col_pv, col_po = st.columns(2)

            with col_pv:
                if "asset" in perps_df.columns and "_volume_usd" in perps_df.columns:
                    pva = perps_df.groupby("asset")["_volume_usd"].sum().sort_values(ascending=False).head(15).reset_index()
                    pva.columns = ["asset", "volume"]
                    pva["volume"] = pva["volume"].round(0)
                    fig = make_h_bar(pva, "volume", "asset", title="Volume by Asset", color="#8B5CF6", x_prefix="$")
                    st.plotly_chart(fig, use_container_width=True)

            with col_po:
                if "asset" in perps_df.columns:
                    poa = perps_df["asset"].value_counts().head(15).reset_index()
                    poa.columns = ["asset", "orders"]
                    fig = make_h_bar(poa, "orders", "asset", title="Orders by Asset", color="#06B6D4")
                    st.plotly_chart(fig, use_container_width=True)

            # Long vs Short + Leverage distribution
            col_ls, col_lev = st.columns(2)

            with col_ls:
                if "side" in perps_df.columns:
                    side_counts = perps_df["side"].value_counts().reset_index()
                    side_counts.columns = ["side", "count"]
                    colors = {"Long": ACCENT, "Short": ACCENT_RED, "long": ACCENT, "short": ACCENT_RED}
                    fig = px.pie(
                        side_counts, values="count", names="side", hole=0.5,
                        color="side", color_discrete_map=colors,
                    )
                    fig.update_traces(textinfo="label+percent+value", textfont=dict(size=13, color="white"))
                    fig.update_layout(**PLOTLY_LAYOUT, title="Long vs Short", height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with col_lev:
                if "leverage" in perps_df.columns:
                    lev_data = pd.to_numeric(perps_df["leverage"], errors="coerce").dropna()
                    if not lev_data.empty:
                        fig = px.histogram(lev_data, nbins=20, title="Leverage Distribution")
                        fig.update_traces(marker_color="#F59E0B")
                        fig.update_layout(**PLOTLY_LAYOUT, height=350)
                        fig.update_xaxes(title="Leverage", **AXIS_DEFAULTS)
                        fig.update_yaxes(title="Count", **AXIS_DEFAULTS)
                        st.plotly_chart(fig, use_container_width=True)

            # Open vs Close — volume & count
            if "is_close" in perps_df.columns:
                order_side = perps_df["is_close"].map(lambda x: "Close" if x else "Open")
                oc_agg = perps_df.assign(order_side=order_side).groupby("order_side").agg(
                    volume=("_volume_usd", "sum"), count=("_volume_usd", "size"),
                ).reset_index()
                oc_agg.columns = ["type", "volume", "count"]

                col_oc_vol, col_oc_cnt = st.columns(2)
                with col_oc_vol:
                    fig = px.bar(oc_agg, x="type", y="volume", title="Open vs Close Volume ($)", color="type",
                                 color_discrete_map={"Open": BRAND, "Close": ACCENT_WARN},
                                 text=[f"${fmt_number(v)}" for v in oc_agg["volume"]])
                    fig.update_traces(textposition="auto", textfont=dict(size=13, color="white"))
                    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
                    fig.update_xaxes(title="")
                    fig.update_yaxes(title="Volume ($)", **AXIS_DEFAULTS)
                    st.plotly_chart(fig, use_container_width=True)

                with col_oc_cnt:
                    fig = px.bar(oc_agg, x="type", y="count", title="Open vs Close Orders", color="type",
                                 color_discrete_map={"Open": BRAND, "Close": ACCENT_WARN})
                    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
                    fig.update_xaxes(title="")
                    fig.update_yaxes(title="Count", **AXIS_DEFAULTS)
                    st.plotly_chart(fig, use_container_width=True)

            # Recent perps table
            st.subheader("Recent Perps Orders")
            perps_cols = [c for c in ["timestamp", "wallet_address", "asset", "side", "size", "price", "leverage", "amount_usd", "order_type", "is_close"] if c in perps_df.columns]
            recent_perps = perps_df[perps_cols].sort_values("timestamp", ascending=False).head(50).copy()
            if "wallet_address" in recent_perps.columns:
                recent_perps["wallet_address"] = recent_perps["wallet_address"].apply(short_wallet)
            if "timestamp" in recent_perps.columns:
                recent_perps["timestamp"] = pd.to_datetime(recent_perps["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%b %d, %I:%M %p")
            st.dataframe(recent_perps, use_container_width=True, hide_index=True)

        # ============================
        # DEPOSITS SECTION
        # ============================
        st.markdown("---")
        st.subheader("Deposits")

        if deposit_df.empty:
            st.info("No deposit data for this period.")
        else:
            dep_total_vol = deposit_df["amount_usd"].sum() if "amount_usd" in deposit_df.columns else 0
            dep_total = len(deposit_df)
            dep_unique = deposit_df["wallet_address"].nunique() if "wallet_address" in deposit_df.columns else 0
            dep_avg = dep_total_vol / dep_total if dep_total > 0 else 0

            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Total Deposits", fmt_number(dep_total))
            dc2.metric("Deposit Volume", f"${fmt_number(dep_total_vol)}")
            dc3.metric("Unique Depositors", fmt_number(dep_unique))
            dc4.metric("Avg Deposit", f"${dep_avg:,.0f}")

            # Deposits by Provider + Daily Deposit Volume
            col_dp, col_dv = st.columns(2)

            with col_dp:
                if "source" in deposit_df.columns:
                    prov = deposit_df["source"].value_counts().reset_index()
                    prov.columns = ["provider", "count"]
                    fig = make_h_bar(prov, "count", "provider", title="Deposits by Provider", color=ACCENT)
                    st.plotly_chart(fig, use_container_width=True)

            with col_dv:
                if "trade_date" in deposit_df.columns and "amount_usd" in deposit_df.columns:
                    daily_dep = deposit_df.groupby("trade_date")["amount_usd"].sum().reset_index()
                    daily_dep.columns = ["date", "volume"]
                    fig = make_time_series(daily_dep, "date", "volume", title="Daily Deposit Volume ($)", color=ACCENT_WARN, kind="bar")
                    st.plotly_chart(fig, use_container_width=True)

            # Recent deposits table
            st.subheader("Recent Deposits")
            dep_cols = [c for c in ["timestamp", "wallet_address", "amount_usd", "source"] if c in deposit_df.columns]
            recent_deps = deposit_df[dep_cols].sort_values("timestamp", ascending=False).head(50).copy()
            if "wallet_address" in recent_deps.columns:
                recent_deps["wallet_address"] = recent_deps["wallet_address"].apply(short_wallet)
            if "timestamp" in recent_deps.columns:
                recent_deps["timestamp"] = pd.to_datetime(recent_deps["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%b %d, %I:%M %p")
            st.dataframe(recent_deps, use_container_width=True, hide_index=True)


# =====================
# TAB 6: Notifications
# =====================
with tab_notifications:
    if df.empty:
        st.info("No data for notification analysis.")
    else:
        # Server-side sends (use full df — these are server events)
        broadcast_events = df[df["event"] == "notification_sent_broadcast"]
        targeted_sends = df[df["event"] == "notification_sent"]
        # Client-side (use user_df — real users only)
        received_events = user_df[user_df["event"] == "notification_received"]
        tapped_events = user_df[user_df["event"] == "notification_tap"]

        # Calculate total server sends
        total_broadcast_sent = 0
        if not broadcast_events.empty and "metadata" in broadcast_events.columns:
            for _, r in broadcast_events.iterrows():
                meta = r.get("metadata")
                if isinstance(meta, dict):
                    total_broadcast_sent += int(meta.get("sent_count", 0))
        total_targeted_sent = len(targeted_sends[targeted_sends.apply(
            lambda r: isinstance(r.get("metadata"), dict) and r["metadata"].get("success", False),
            axis=1,
        )]) if not targeted_sends.empty and "metadata" in targeted_sends.columns else 0
        total_server_sent = total_broadcast_sent + total_targeted_sent

        received_count = len(received_events)
        tapped_count = len(tapped_events)
        tap_rate_server = (tapped_count / total_server_sent * 100) if total_server_sent > 0 else 0
        tap_rate_client = (tapped_count / received_count * 100) if received_count > 0 else 0
        unique_tappers = tapped_events["wallet_address"].nunique()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sent (server)", fmt_number(total_server_sent))
        col2.metric("Received (client)", fmt_number(received_count))
        col3.metric("Tapped", fmt_number(tapped_count))
        col4.metric("Tap Rate (server)", f"{tap_rate_server:.1f}%")
        col5.metric("Unique Tappers", fmt_number(unique_tappers))

        # Breakdown by notification type
        if not broadcast_events.empty or not targeted_sends.empty:
            st.markdown("---")
            st.subheader("Sends by Type")
            type_data = []
            if not broadcast_events.empty and "metadata" in broadcast_events.columns:
                type_data.append({"Type": "Trade Alerts (broadcast)", "Sends": len(broadcast_events), "Recipients": total_broadcast_sent})
            if not targeted_sends.empty and "metadata" in targeted_sends.columns:
                for ntype in ["price_alert", "lifecycle"]:
                    typed = targeted_sends[targeted_sends.apply(
                        lambda r: isinstance(r.get("metadata"), dict) and r["metadata"].get("type") == ntype, axis=1
                    )]
                    if len(typed) > 0:
                        type_data.append({"Type": ntype.replace("_", " ").title(), "Sends": len(typed), "Recipients": len(typed)})
            if type_data:
                st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Tap rate by day
        if not received_events.empty:
            st.subheader("Tap Rate by Day")
            daily_recv = received_events.groupby("date").size()
            daily_tap = tapped_events.groupby("date").size() if not tapped_events.empty else pd.Series(dtype=int)
            daily_rate = ((daily_tap / daily_recv * 100).fillna(0)).reset_index()
            daily_rate.columns = ["date", "rate"]
            fig = make_time_series(daily_rate, "date", "rate", title="Tap Rate %", color=ACCENT)
            fig.update_yaxes(title="Tap Rate %")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Time to trade
        st.subheader("Time to Trade After Notification Tap")
        if not tapped_events.empty and "ts" in user_df.columns:
            trade_events = user_df[user_df["event"] == "trade_initiated"].copy()
            if not trade_events.empty:
                times_to_trade = []
                for _, tap in tapped_events.iterrows():
                    wallet = tap["wallet_address"]
                    tap_ts = tap.get("ts")
                    if pd.isna(tap_ts):
                        continue
                    wallet_trades = trade_events[
                        (trade_events["wallet_address"] == wallet)
                        & (trade_events["ts"] > tap_ts)
                        & (trade_events["ts"] < tap_ts + timedelta(hours=1))
                    ]
                    if not wallet_trades.empty:
                        delta = (wallet_trades.iloc[0]["ts"] - tap_ts).total_seconds()
                        times_to_trade.append(delta)

                if times_to_trade:
                    avg_s = sum(times_to_trade) / len(times_to_trade)
                    med_s = sorted(times_to_trade)[len(times_to_trade) // 2]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Time", f"{avg_s:.0f}s")
                    col2.metric("Median Time", f"{med_s:.0f}s")
                    col3.metric("Conversions (<1hr)", fmt_number(len(times_to_trade)))

                    # Distribution
                    ttd_df = pd.DataFrame({"seconds": times_to_trade})
                    fig = px.histogram(ttd_df, x="seconds", nbins=20, title="Time to Trade Distribution")
                    fig.update_traces(marker_color=BRAND)
                    fig.update_layout(**PLOTLY_LAYOUT, height=300)
                    fig.update_xaxes(title="Seconds after tap")
                    fig.update_yaxes(title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No notification→trade conversions found within 1 hour.")
            else:
                st.info("No trade events found.")
        else:
            st.info("No notification tap events found.")

        st.markdown("---")

        # Top tapped tickers + Top tappers side by side
        col_tick, col_tap = st.columns(2)

        with col_tick:
            if not tapped_events.empty and "metadata" in tapped_events.columns:
                st.subheader("Most Tapped Tickers")
                tickers = []
                for _, row in tapped_events.iterrows():
                    meta = row.get("metadata")
                    if isinstance(meta, dict) and meta.get("ticker"):
                        tickers.append(meta["ticker"])
                if tickers:
                    tc = pd.Series(tickers).value_counts().head(10).reset_index()
                    tc.columns = ["ticker", "taps"]
                    fig = make_h_bar(tc, "taps", "ticker", title="Top Tickers", color=ACCENT_WARN)
                    st.plotly_chart(fig, use_container_width=True)

        with col_tap:
            if not tapped_events.empty:
                st.subheader("Most Engaged Users")
                tt = tapped_events.groupby("wallet_address").size().sort_values(ascending=False).head(10).reset_index()
                tt.columns = ["wallet_address", "taps"]
                tt["wallet"] = tt["wallet_address"].apply(short_wallet)
                fig = make_h_bar(tt, "taps", "wallet", title="Top Notification Tappers", color="#EC4899")
                st.plotly_chart(fig, use_container_width=True)


# =====================
# TAB 7: Backend Health
# =====================
with tab_backend:
    st.subheader("Trading Backend — CloudWatch Metrics")

    bh_hours = st.selectbox("Lookback", [6, 12, 24, 48, 72, 168], index=2, key="bh_hours",
                             format_func=lambda h: f"{h}h" if h < 168 else "7 days")
    bh_period = 300 if bh_hours <= 24 else (900 if bh_hours <= 72 else 3600)

    # --- Fee Payer Balance (critical) — live RPC ---
    st.markdown("### Gas Sponsorship")
    evm_bal = fetch_evm_balance(EVM_FEE_PAYER)
    sol_bal = fetch_sol_balance(SOL_FEE_PAYER)

    col_evm, col_sol = st.columns(2)
    if evm_bal is not None:
        evm_low = evm_bal < 0.005
        col_evm.metric("EVM Fee Payer (Arbitrum)", f"{evm_bal:.6f} ETH",
                        delta="LOW" if evm_low else "OK",
                        delta_color="inverse" if evm_low else "normal")
    else:
        col_evm.metric("EVM Fee Payer (Arbitrum)", "RPC error")

    if sol_bal is not None:
        sol_low = sol_bal < 0.5
        col_sol.metric("SOL Fee Payer", f"{sol_bal:.4f} SOL",
                        delta="LOW" if sol_low else "OK",
                        delta_color="inverse" if sol_low else "normal")
    else:
        col_sol.metric("SOL Fee Payer", "RPC error")

    # CloudWatch history (bonus — shows if data exists)
    fee_df = load_cw_metric("Gas.FeePayerBalanceETH", stat="Minimum", period=bh_period, hours=bh_hours)
    if not fee_df.empty:
        fig = make_time_series(fee_df, "timestamp", "value", title="Fee Payer ETH Balance (CloudWatch)", color=ACCENT_WARN, kind="area")
        fig.add_hline(y=0.005, line_dash="dash", line_color=ACCENT_RED, annotation_text="Alarm threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- API Health ---
    st.markdown("### API Health")
    api_latency = load_cw_metric("API.Latency", stat="Average", period=bh_period, hours=bh_hours)
    api_requests = load_cw_metric("API.Request", stat="Sum", period=bh_period, hours=bh_hours)
    api_errors = load_cw_metric("API.Error", stat="Sum", period=bh_period, hours=bh_hours)

    col_req, col_lat, col_err = st.columns(3)
    total_reqs = int(api_requests["value"].sum()) if not api_requests.empty else 0
    avg_lat = api_latency["value"].mean() if not api_latency.empty else 0
    total_errs = int(api_errors["value"].sum()) if not api_errors.empty else 0
    col_req.metric("Total Requests", fmt_number(total_reqs))
    col_lat.metric("Avg Latency", f"{avg_lat:.0f} ms")
    col_err.metric("Errors (4xx+5xx)", fmt_number(total_errs))

    if not api_latency.empty:
        fig = make_time_series(api_latency, "timestamp", "value", title="API Latency (ms)", color=BRAND, kind="area")
        st.plotly_chart(fig, use_container_width=True)

    if not api_requests.empty:
        fig = make_time_series(api_requests, "timestamp", "value", title="Request Volume", color=ACCENT, kind="bar")
        st.plotly_chart(fig, use_container_width=True)

    if not api_errors.empty and total_errs > 0:
        fig = make_time_series(api_errors, "timestamp", "value", title="API Errors", color=ACCENT_RED, kind="bar")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Order Latency by Venue ---
    st.markdown("### Order Execution")
    venues = ["hyperliquid", "ostium"]
    order_lat_df = load_cw_metric_by_dims("Order.PlaceLatency", "Venue", venues, stat="Average", period=bh_period, hours=bh_hours)

    if not order_lat_df.empty:
        col_hl, col_os = st.columns(2)
        for col, venue in [(col_hl, "hyperliquid"), (col_os, "ostium")]:
            vdf = order_lat_df[order_lat_df["dimension"] == venue]
            avg = vdf["value"].mean() if not vdf.empty else 0
            col.metric(f"{venue.title()} Avg Latency", f"{avg:.0f} ms")
        fig = px.line(order_lat_df, x="timestamp", y="value", color="dimension",
                      title="Order Placement Latency by Venue (ms)",
                      color_discrete_sequence=[BRAND, ACCENT])
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        fig.update_xaxes(title="", tickformat="%b %d %H:%M", **AXIS_DEFAULTS)
        fig.update_yaxes(title="ms", **AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    cancel_lat = load_cw_metric("Order.CancelLatency", stat="Average", period=bh_period, hours=bh_hours)
    withdraw_lat = load_cw_metric("Withdraw.Latency", stat="Average", period=bh_period, hours=bh_hours)

    col_cl, col_wl = st.columns(2)
    col_cl.metric("Avg Cancel Latency", f"{cancel_lat['value'].mean():.0f} ms" if not cancel_lat.empty else "N/A")
    col_wl.metric("Avg Withdraw Latency", f"{withdraw_lat['value'].mean():.0f} ms" if not withdraw_lat.empty else "N/A")

    st.markdown("---")

    # --- Bridge Orchestrator ---
    st.markdown("### Bridge Orchestrator")
    bridge_success = load_cw_metric("Bridge.Success", stat="Sum", period=bh_period, hours=bh_hours)
    bridge_fail = load_cw_metric("Bridge.Failure", stat="Sum", period=bh_period, hours=bh_hours)
    bridge_dur = load_cw_metric("Bridge.TotalDuration", stat="Average", period=bh_period, hours=bh_hours)
    bridge_fill = load_cw_metric("Bridge.FillTime", stat="Average", period=bh_period, hours=bh_hours)

    col_bs, col_bf, col_bd, col_bfi = st.columns(4)
    s_total = int(bridge_success["value"].sum()) if not bridge_success.empty else 0
    f_total = int(bridge_fail["value"].sum()) if not bridge_fail.empty else 0
    col_bs.metric("Successful Bridges", s_total)
    col_bf.metric("Failed Bridges", f_total)
    avg_dur = bridge_dur["value"].mean() / 1000 if not bridge_dur.empty else 0
    col_bd.metric("Avg Duration", f"{avg_dur:.0f}s")
    avg_fill = bridge_fill["value"].mean() / 1000 if not bridge_fill.empty else 0
    col_bfi.metric("Avg Fill Time", f"{avg_fill:.0f}s")

    if not bridge_dur.empty:
        bridge_dur_s = bridge_dur.copy()
        bridge_dur_s["value"] = bridge_dur_s["value"] / 1000
        fig = make_time_series(bridge_dur_s, "timestamp", "value", title="Bridge Total Duration (seconds)", color=BRAND_LIGHT, kind="area")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Fund Router ---
    st.markdown("### Fund Router (Arb <> HL)")
    route_lat = load_cw_metric("FundRouter.RouteLatency", stat="Average", period=bh_period, hours=bh_hours)
    rev_lat = load_cw_metric("FundRouter.ReverseRouteLatency", stat="Average", period=bh_period, hours=bh_hours)
    settle_timeout = load_cw_metric("FundRouter.SettlementTimeout", stat="Sum", period=bh_period, hours=bh_hours)

    col_rl2, col_rr, col_st2 = st.columns(3)
    col_rl2.metric("Avg Route Latency (Arb>HL)", f"{route_lat['value'].mean() / 1000:.1f}s" if not route_lat.empty else "N/A")
    col_rr.metric("Avg Reverse Route (HL>Arb)", f"{rev_lat['value'].mean() / 1000:.1f}s" if not rev_lat.empty else "N/A")
    col_st2.metric("Settlement Timeouts", f"{int(settle_timeout['value'].sum())}" if not settle_timeout.empty else "0")

    if not route_lat.empty:
        route_lat_s = route_lat.copy()
        route_lat_s["value"] = route_lat_s["value"] / 1000
        fig = make_time_series(route_lat_s, "timestamp", "value", title="Fund Route Latency (seconds)", color=ACCENT, kind="area")
        st.plotly_chart(fig, use_container_width=True)

    # --- Venue Read Performance ---
    st.markdown("---")
    st.markdown("### Venue Read Performance")
    venue_read_df = load_cw_metric_by_dims("Venue.ReadLatency", "Venue", ["hyperliquid", "lighter", "ostium"],
                                            stat="Average", period=bh_period, hours=bh_hours)
    venue_err_df = load_cw_metric_by_dims("Venue.ReadError", "Venue", ["hyperliquid", "lighter", "ostium"],
                                           stat="Sum", period=bh_period, hours=bh_hours)

    if not venue_read_df.empty:
        col_v1, col_v2, col_v3 = st.columns(3)
        for col, venue in [(col_v1, "hyperliquid"), (col_v2, "lighter"), (col_v3, "ostium")]:
            vdf = venue_read_df[venue_read_df["dimension"] == venue]
            avg = vdf["value"].mean() if not vdf.empty else 0
            edf = venue_err_df[venue_err_df["dimension"] == venue] if not venue_err_df.empty else pd.DataFrame()
            errs = int(edf["value"].sum()) if not edf.empty else 0
            col.metric(f"{venue.title()} Read", f"{avg:.0f} ms", delta=f"{errs} errors" if errs > 0 else None,
                       delta_color="inverse" if errs > 0 else "off")
        fig = px.line(venue_read_df, x="timestamp", y="value", color="dimension",
                      title="Venue Read Latency (ms)", color_discrete_sequence=[BRAND, ACCENT, ACCENT_WARN])
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        fig.update_xaxes(title="", tickformat="%b %d %H:%M", **AXIS_DEFAULTS)
        fig.update_yaxes(title="ms", **AXIS_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    # --- Cache Performance ---
    st.markdown("---")
    st.markdown("### Cache Performance")
    cache_hit = load_cw_metric("Cache.Hit", stat="Sum", period=bh_period, hours=bh_hours)
    cache_miss = load_cw_metric("Cache.Miss", stat="Sum", period=bh_period, hours=bh_hours)

    hits = int(cache_hit["value"].sum()) if not cache_hit.empty else 0
    misses = int(cache_miss["value"].sum()) if not cache_miss.empty else 0
    total_cache = hits + misses
    hit_rate = (hits / total_cache * 100) if total_cache > 0 else 0

    col_hr, col_h, col_m = st.columns(3)
    col_hr.metric("Hit Rate", f"{hit_rate:.1f}%")
    col_h.metric("Cache Hits", fmt_number(hits))
    col_m.metric("Cache Misses", fmt_number(misses))

    if not cache_hit.empty or not cache_miss.empty:
        cache_frames = []
        if not cache_hit.empty:
            h = cache_hit.copy()
            h["type"] = "Hit"
            cache_frames.append(h)
        if not cache_miss.empty:
            m = cache_miss.copy()
            m["type"] = "Miss"
            cache_frames.append(m)
        if cache_frames:
            cdf = pd.concat(cache_frames, ignore_index=True)
            fig = px.bar(cdf, x="timestamp", y="value", color="type",
                         title="Cache Hits vs Misses", color_discrete_sequence=[ACCENT, ACCENT_RED], barmode="stack")
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            fig.update_xaxes(title="", tickformat="%b %d %H:%M", **AXIS_DEFAULTS)
            fig.update_yaxes(title="", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    # --- WebSocket ---
    st.markdown("---")
    st.markdown("### WebSocket Fan-out")
    ws_conns = load_cw_metric("WS.ActiveConnections", stat="Maximum", period=bh_period, hours=bh_hours)
    ws_sent = load_cw_metric("WS.MessageSent", stat="Sum", period=bh_period, hours=bh_hours)
    ws_recv = load_cw_metric("WS.MessageReceived", stat="Sum", period=bh_period, hours=bh_hours)
    ws_slow = load_cw_metric("WS.SlowClientTerminated", stat="Sum", period=bh_period, hours=bh_hours)

    col_wc, col_ws, col_wr, col_wsl = st.columns(4)
    col_wc.metric("Peak Connections", f"{int(ws_conns['value'].max())}" if not ws_conns.empty else "0")
    col_ws.metric("Messages Sent", fmt_number(int(ws_sent["value"].sum())) if not ws_sent.empty else "0")
    col_wr.metric("Messages Received", fmt_number(int(ws_recv["value"].sum())) if not ws_recv.empty else "0")
    slow_count = int(ws_slow["value"].sum()) if not ws_slow.empty else 0
    col_wsl.metric("Slow Clients Killed", str(slow_count), delta=f"{slow_count}" if slow_count > 0 else None,
                   delta_color="inverse")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if not ws_conns.empty:
            fig = make_time_series(ws_conns, "timestamp", "value", title="Active WS Connections", color=BRAND_LIGHT, kind="area")
            st.plotly_chart(fig, use_container_width=True)
    with col_chart2:
        if not ws_sent.empty:
            fig = make_time_series(ws_sent, "timestamp", "value", title="Messages Sent (fan-out throughput)", color=ACCENT, kind="area")
            st.plotly_chart(fig, use_container_width=True)

    # Capacity indicator
    if not ws_conns.empty:
        peak = int(ws_conns["value"].max())
        capacity = 6 * 3500  # max_tasks * approx connections per task
        pct = (peak / capacity) * 100 if capacity > 0 else 0
        st.progress(min(pct / 100, 1.0), text=f"Capacity: {peak:,} / {capacity:,} ({pct:.1f}%) — auto-scales min 2, max 6 tasks")

    # --- Rate Limiting & Request Health ---
    st.markdown("---")
    st.markdown("### Rate Limiting & Request Health")
    st.caption("REST rate limit: 100 req/min per IP. Throttled requests return 429 and appear in the error count.")
    api_total = load_cw_metric("API.Request", stat="Sum", period=bh_period, hours=bh_hours)
    api_err = load_cw_metric("API.Error", stat="Sum", period=bh_period, hours=bh_hours)

    col_rl1, col_rl2, col_rl3 = st.columns(3)
    total_req = int(api_total["value"].sum()) if not api_total.empty else 0
    total_err = int(api_err["value"].sum()) if not api_err.empty else 0
    err_rate = (total_err / total_req * 100) if total_req > 0 else 0
    col_rl1.metric("Total Requests", fmt_number(total_req))
    col_rl2.metric("Total Errors (4xx+5xx)", fmt_number(total_err))
    col_rl3.metric("Error Rate", f"{err_rate:.2f}%",
                   delta=f"{err_rate:.2f}%" if err_rate > 1 else None, delta_color="inverse")

    if not api_err.empty and api_err["value"].sum() > 0:
        fig = make_time_series(api_err, "timestamp", "value", title="API Errors Over Time (includes 429 rate limits)", color=ACCENT_RED, kind="bar")
        st.plotly_chart(fig, use_container_width=True)

    # --- Privy Signing ---
    st.markdown("---")
    st.markdown("### Privy Signing")
    privy_lat = load_cw_metric("Privy.SigningLatency", stat="Average", period=bh_period, hours=bh_hours)

    col_pl, col_pc = st.columns(2)
    col_pl.metric("Avg Signing Latency", f"{privy_lat['value'].mean():.0f} ms" if not privy_lat.empty else "N/A")
    col_pc.metric("Total Signings", f"{len(privy_lat)}" if not privy_lat.empty else "0")

    if not privy_lat.empty:
        fig = make_time_series(privy_lat, "timestamp", "value", title="Privy Signing Latency (ms)", color="#8B5CF6", kind="area")
        st.plotly_chart(fig, use_container_width=True)

    # --- Account Snapshot ---
    st.markdown("---")
    st.markdown("### Account Snapshots")
    acct_lat = load_cw_metric("Account.SnapshotLatency", stat="Average", period=bh_period, hours=bh_hours)

    if not acct_lat.empty:
        st.metric("Avg Multi-Venue Snapshot", f"{acct_lat['value'].mean():.0f} ms")
        fig = make_time_series(acct_lat, "timestamp", "value", title="Account Snapshot Latency (ms)", color="#06B6D4", kind="area")
        st.plotly_chart(fig, use_container_width=True)

    # --- Alarms status ---
    st.markdown("---")
    st.markdown("### Alarm Status")
    st.caption("CloudWatch alarms for: CPU high (>80%), WS connections high (>3000), transfer queue backlog (>300s), fee payer balance low (<0.005 ETH)")
    try:
        cw = get_cloudwatch()
        # Fetch all alarms (not just firing) for a full overview
        all_alarms_resp = cw.describe_alarms()
        all_alarms = all_alarms_resp.get("MetricAlarms", [])
        firing = [a for a in all_alarms if a["StateValue"] == "ALARM"]
        ok_alarms = [a for a in all_alarms if a["StateValue"] == "OK"]

        if firing:
            for alarm in firing:
                st.error(f"🔴 **{alarm['AlarmName']}** — {alarm.get('AlarmDescription', 'No description')}")
        if ok_alarms:
            alarm_names = ", ".join(a["AlarmName"].split("-")[-1] for a in ok_alarms)
            st.success(f"All {len(ok_alarms)} alarms OK: {alarm_names}")
        if not all_alarms:
            st.info("No alarms configured")
    except Exception as e:
        st.warning(f"Could not fetch alarms: {e}")


# =====================
# TAB 8: Services Health (Render + CloudFront + Logs)
# =====================
with tab_services:
    st.subheader("Infrastructure Health")

    sh_hours = st.selectbox("Lookback", [1, 6, 12, 24, 48, 72, 168], index=3, key="sh_hours",
                             format_func=lambda h: f"{h}h" if h < 168 else "7 days")
    sh_period = 300 if sh_hours <= 24 else (900 if sh_hours <= 72 else 3600)

    # --- Discover services ---
    try:
        cw = get_cloudwatch()
        all_svc_metrics = cw.list_metrics(Namespace=RENDER_CW_NAMESPACE, MetricName="render.service.memory.usage")
        discovered_services = sorted(set(
            d["Value"] for m in all_svc_metrics.get("Metrics", [])
            for d in m.get("Dimensions", []) if d["Name"] == "service.name"
        ))
    except Exception:
        discovered_services = []
    if not discovered_services:
        discovered_services = RENDER_SERVICES

    # --- Load metrics ---
    mem_df = load_render_metric_by_service("render.service.memory.usage", stat="Average", period=sh_period, hours=sh_hours)
    cpu_df = load_render_metric_by_service("render.service.cpu.time", stat="Average", period=sh_period, hours=sh_hours)
    http_df = load_render_metric_by_service("render.service.http.requests.total", stat="Sum", period=sh_period, hours=sh_hours)
    net_tx_df = load_render_metric_by_service("render.service.network.transmit.bytes", stat="Sum", period=sh_period, hours=sh_hours)
    latency_df = load_render_metric_by_service("render.service.http.requests.latency", stat="Average", period=sh_period, hours=sh_hours)
    mem_limit_df = load_render_metric_by_service("render.service.memory.limit", stat="Maximum", period=sh_period, hours=sh_hours)

    has_metrics = not mem_df.empty or not cpu_df.empty or not http_df.empty

    # ============================================================
    # SECTION 1: Service Health Status (at-a-glance)
    # ============================================================
    st.markdown("### Service Health Status")

    if not has_metrics:
        st.info(
            "No Render metrics yet. Ensure Render Metrics Stream is configured:\n\n"
            "- **Provider**: Custom | **Endpoint**: `https://otel.freeportmarkets.com` | **Token**: any value\n\n"
            "Metrics appear within ~5 minutes."
        )
    else:
        # Build health status per service
        svc_cols = st.columns(min(len(discovered_services), 5))
        for i, svc in enumerate(discovered_services[:5]):
            svc_mem = mem_df[mem_df["service"] == svc] if not mem_df.empty else pd.DataFrame()
            svc_limit = mem_limit_df[mem_limit_df["service"] == svc] if not mem_limit_df.empty else pd.DataFrame()
            svc_http = http_df[http_df["service"] == svc] if not http_df.empty else pd.DataFrame()

            avg_mem = svc_mem["value"].mean() if not svc_mem.empty else 0
            mem_limit = svc_limit["value"].max() if not svc_limit.empty else 0
            mem_pct = (avg_mem / mem_limit * 100) if mem_limit > 0 else 0
            total_reqs = int(svc_http["value"].sum()) if not svc_http.empty else 0
            has_recent = not svc_mem.empty and len(svc_mem) > 2  # getting data points

            # Determine health color
            if not has_recent:
                status, color = "NO DATA", "#6B7280"
            elif mem_pct > 85:
                status, color = "CRITICAL", ACCENT_RED
            elif mem_pct > 70:
                status, color = "WARNING", ACCENT_WARN
            else:
                status, color = "HEALTHY", ACCENT

            label = svc.replace("-", " ").title()
            with svc_cols[i]:
                st.markdown(
                    f'<div style="border:2px solid {color};border-radius:12px;padding:12px;text-align:center;">'
                    f'<div style="color:{color};font-weight:700;font-size:0.8rem;letter-spacing:0.05em;">{status}</div>'
                    f'<div style="font-size:1.1rem;font-weight:600;margin:4px 0;">{label}</div>'
                    f'<div style="color:{TEXT_MUTED};font-size:0.85rem;">'
                    f'{avg_mem / (1024*1024):.0f} MB ({mem_pct:.0f}% limit)<br>'
                    f'{fmt_number(total_reqs)} requests</div></div>',
                    unsafe_allow_html=True,
                )

        # Error rate from logs (quick Insights query)
        error_stats = query_render_logs_stats(
            'filter @message like /(?i)(error|exception|fatal|crash|ECONNREFUSED)/'
            ' | stats count(*) as errors by bin(1h) as hour'
            ' | sort hour desc',
            hours=sh_hours if sh_hours <= 24 else 24,
        )
        if not error_stats.empty and "errors" in error_stats.columns:
            total_errors = pd.to_numeric(error_stats["errors"], errors="coerce").sum()
            if total_errors > 0:
                st.warning(f"**{int(total_errors)} errors** detected across all services in the last {min(sh_hours, 24)}h")

    # ============================================================
    # SECTION 2: Render Service Metrics (visual charts)
    # ============================================================
    if has_metrics:
        st.markdown("---")
        st.markdown("### Render Service Metrics")

        # --- Memory Usage with limit lines ---
        if not mem_df.empty:
            mem_df["value_mb"] = mem_df["value"] / (1024 * 1024)
            fig = go.Figure()
            colors = {svc: CHART_COLORS[i % len(CHART_COLORS)] for i, svc in enumerate(mem_df["service"].unique())}
            for svc in mem_df["service"].unique():
                sdf = mem_df[mem_df["service"] == svc]
                fig.add_trace(go.Scatter(
                    x=sdf["timestamp"], y=sdf["value_mb"], name=svc,
                    mode="lines", fill="tozeroy",
                    line=dict(color=colors[svc], width=2),
                    fillcolor=f"rgba({int(colors[svc][1:3],16)},{int(colors[svc][3:5],16)},{int(colors[svc][5:7],16)},0.1)",
                ))
            # Add limit lines if available
            if not mem_limit_df.empty:
                for svc in mem_limit_df["service"].unique():
                    lim = mem_limit_df[mem_limit_df["service"] == svc]["value"].max() / (1024 * 1024)
                    fig.add_hline(y=lim, line_dash="dot", line_color="rgba(239,68,68,0.4)",
                                 annotation_text=f"{svc} limit", annotation_font_size=10)
            fig.update_layout(**PLOTLY_LAYOUT, title="Memory Usage by Service (MB)", height=350,
                              legend=dict(orientation="h", y=-0.2))
            fig.update_xaxes(title="", tickformat="%b %d %H:%M", **AXIS_DEFAULTS)
            fig.update_yaxes(title="MB", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

        # --- CPU + HTTP side by side ---
        col_cpu, col_http = st.columns(2)
        with col_cpu:
            if not cpu_df.empty:
                fig = px.line(cpu_df, x="timestamp", y="value", color="service",
                              title="CPU Time by Service", color_discrete_sequence=CHART_COLORS)
                fig.update_layout(**PLOTLY_LAYOUT, height=300, legend=dict(orientation="h", y=-0.25))
                fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
                fig.update_yaxes(title="seconds", **AXIS_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True)

        with col_http:
            if not http_df.empty:
                fig = px.bar(http_df, x="timestamp", y="value", color="service",
                             title="HTTP Requests by Service", color_discrete_sequence=CHART_COLORS, barmode="stack")
                fig.update_layout(**PLOTLY_LAYOUT, height=300, legend=dict(orientation="h", y=-0.25))
                fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
                fig.update_yaxes(title="requests", **AXIS_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True)

        # --- Latency + Network side by side ---
        col_lat, col_net = st.columns(2)
        with col_lat:
            if not latency_df.empty:
                fig = px.line(latency_df, x="timestamp", y="value", color="service",
                              title="HTTP Latency by Service", color_discrete_sequence=CHART_COLORS)
                fig.update_layout(**PLOTLY_LAYOUT, height=300, legend=dict(orientation="h", y=-0.25))
                fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
                fig.update_yaxes(title="ms", **AXIS_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True)

        with col_net:
            if not net_tx_df.empty:
                net_tx_df["value_kb"] = net_tx_df["value"] / 1024
                fig = px.area(net_tx_df, x="timestamp", y="value_kb", color="service",
                              title="Network Transmit by Service (KB)", color_discrete_sequence=CHART_COLORS)
                fig.update_layout(**PLOTLY_LAYOUT, height=300, legend=dict(orientation="h", y=-0.25))
                fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
                fig.update_yaxes(title="KB", **AXIS_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # SECTION 3: Error Dashboard
    # ============================================================
    # Render syslog priorities: <3>=error, <4>=warning (HTTP 4xx), <6>=info, <7>=debug
    # Swap server HTTP errors arrive at priority <4> with JSON body containing "statusCode":4xx
    # App-level errors (CoinGecko 429, connection pool) arrive at <3> or <4> as plain text
    st.markdown("---")
    st.markdown("### Error Dashboard")
    st.caption("Syslog priority `<3>` (error) + `<4>` (warning/HTTP 4xx) from all Render services")

    insight_hours = min(sh_hours, 24)  # Insights queries capped at 24h for speed

    # --- Errors by service (priority <= 4 catches errors + warnings + HTTP 4xx) ---
    err_by_svc = query_render_logs_stats(
        'parse @message /^<(?<pri>\\d+)>/'
        ' | filter pri <= 4'
        ' | parse @message /^<\\d+>\\d+ \\S+ (?<svc>\\S+)/'
        ' | stats count(*) as errors by svc'
        ' | sort errors desc'
        ' | limit 15',
        hours=insight_hours,
    )

    # --- Error timeline by hour ---
    err_timeline = query_render_logs_stats(
        'parse @message /^<(?<pri>\\d+)>/'
        ' | filter pri <= 4'
        ' | stats count(*) as errors by bin(1h) as hour'
        ' | sort hour asc',
        hours=insight_hours,
    )

    # --- Top error messages ---
    # Two log formats:
    #   1. HTTP access logs: <4>1 TS svc http-request ... - {"statusCode":400,"path":"/order?..."}
    #   2. App logs: <3>1 TS svc proc ... - CoinGecko API error: 429
    # Query them separately and merge for a clean table.
    err_http = query_render_logs_stats(
        'parse @message /^<(?<pri>\\d+)>/'
        ' | filter pri <= 4 and @message like /http-request/'
        ' | parse @message /^<\\d+>\\d+ \\S+ (?<svc>\\S+)/'
        ' | parse @message /^<\\d+>\\d+ \\S+ \\S+ \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | parse body /"statusCode":(?<sc>\\d+)/'
        ' | filter sc >= 400'
        ' | parse body /"path":"(?<rawpath>[^"]+)"/'
        ' | parse rawpath /^\\/(?<endpoint>[^?]+)/'
        ' | stats count(*) as cnt by svc, concat("HTTP ", sc, " /", endpoint) as detail'
        ' | sort cnt desc'
        ' | limit 15',
        hours=insight_hours,
    )
    err_app = query_render_logs_stats(
        'parse @message /^<(?<pri>\\d+)>/'
        ' | filter pri <= 4 and @message not like /http-request/'
        ' | parse @message /^<\\d+>\\d+ \\S+ (?<svc>\\S+) \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | stats count(*) as cnt by svc, body as detail'
        ' | sort cnt desc'
        ' | limit 15',
        hours=insight_hours,
    )
    # Merge HTTP + app errors into one table
    err_messages = pd.concat([err_http, err_app], ignore_index=True)
    if not err_messages.empty and "cnt" in err_messages.columns:
        err_messages["cnt"] = pd.to_numeric(err_messages["cnt"], errors="coerce")
        err_messages = err_messages.sort_values("cnt", ascending=False).head(25)

    # --- Recent errors (individual entries) ---
    recent_errors = query_render_logs_stats(
        'parse @message /^<(?<pri>\\d+)>/'
        ' | filter pri <= 4'
        ' | parse @message /^<\\d+>\\d+ (?<ts>\\S+) (?<svc>\\S+) \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | fields ts, svc, body'
        ' | sort @timestamp desc'
        ' | limit 30',
        hours=insight_hours,
    )

    # --- Summary banner ---
    total_errors = 0
    if not err_by_svc.empty and "errors" in err_by_svc.columns:
        err_by_svc["errors"] = pd.to_numeric(err_by_svc["errors"], errors="coerce")
        total_errors = int(err_by_svc["errors"].sum())

    if total_errors > 0:
        top_svc = err_by_svc.iloc[0]["svc"] if "svc" in err_by_svc.columns else "unknown"
        top_cnt = int(err_by_svc.iloc[0]["errors"])
        st.error(f"**{total_errors} errors** in the last {insight_hours}h — most from **{top_svc}** ({top_cnt})")
    else:
        st.success(f"No errors in the last {insight_hours}h")

    # --- Charts row 1: Error timeline + Errors by service ---
    col_timeline, col_svc = st.columns(2)

    with col_timeline:
        if not err_timeline.empty and "errors" in err_timeline.columns:
            err_timeline["errors"] = pd.to_numeric(err_timeline["errors"], errors="coerce")
            if "hour" in err_timeline.columns:
                err_timeline["hour"] = pd.to_datetime(err_timeline["hour"], errors="coerce")
            fig = go.Figure(go.Bar(
                x=err_timeline.get("hour", err_timeline.index),
                y=err_timeline["errors"],
                marker_color=ACCENT_RED,
                hovertemplate="<b>%{x}</b><br>%{y} errors<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Errors per Hour", height=300)
            fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
            fig.update_yaxes(title="errors", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    with col_svc:
        if not err_by_svc.empty and "svc" in err_by_svc.columns:
            fig = go.Figure(go.Bar(
                y=err_by_svc["svc"], x=err_by_svc["errors"],
                orientation="h", marker_color=ACCENT_RED,
                text=[fmt_number(v) for v in err_by_svc["errors"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} errors<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Errors by Service", height=300,
                              yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(title="", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    # --- Top error messages table ---
    if not err_messages.empty and "cnt" in err_messages.columns:
        st.markdown("#### Top Error Messages")
        err_messages["cnt"] = pd.to_numeric(err_messages["cnt"], errors="coerce")
        if "detail" in err_messages.columns:
            err_messages["detail"] = err_messages["detail"].fillna("(unstructured)")
            # Truncate long query strings for readability
            err_messages["detail"] = err_messages["detail"].apply(
                lambda x: (x[:120] + "...") if isinstance(x, str) and len(x) > 120 else x
            )
        display_cols = [c for c in ["svc", "detail", "cnt"] if c in err_messages.columns]
        display_df = err_messages[display_cols].rename(
            columns={"svc": "Service", "detail": "Error Detail", "cnt": "Count"}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

    # --- Recent errors table ---
    if not recent_errors.empty:
        st.markdown("#### Recent Errors")
        display_cols = [c for c in ["ts", "svc", "body"] if c in recent_errors.columns]
        if display_cols:
            recent_display = recent_errors[display_cols].rename(
                columns={"ts": "Time", "svc": "Service", "body": "Error Details"}
            )
            st.dataframe(recent_display, use_container_width=True, hide_index=True, height=400)

    # --- Swap server specific: HTTP 4xx/5xx from structured access logs ---
    # Actual format: <4>1 TIMESTAMP swap-server-1 http-request 1 http-request - {"statusCode":400,"path":"/order?inputMint=...&outputMint=...","method":"GET",...}
    # Also catches 499 (client disconnect during /execute) and other non-200 responses
    st.markdown("---")
    st.markdown("### Swap Server — Failed Requests")
    st.caption("HTTP 4xx/5xx from swap-server-1 access logs (parsed from JSON `statusCode` field)")

    # --- Swap errors by status code + endpoint ---
    swap_errors = query_render_logs_stats(
        'filter @message like /swap-server/ and @message like /http-request/'
        ' | parse @message /^<\\d+>\\d+ \\S+ \\S+ \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | parse body /"statusCode":(?<sc>\\d+)/'
        ' | filter sc >= 400'
        ' | parse body /"path":"(?<rawpath>[^"]+)"/'
        ' | parse rawpath /^\\/(?<endpoint>[^?]+)/'
        ' | parse rawpath /outputMint=(?<out_mint>[^&]+)/'
        ' | stats count(*) as cnt by sc, endpoint, out_mint'
        ' | sort cnt desc'
        ' | limit 20',
        hours=insight_hours,
    )

    # --- Timeline ---
    swap_errors_timeline = query_render_logs_stats(
        'filter @message like /swap-server/ and @message like /http-request/'
        ' | parse @message /^<\\d+>\\d+ \\S+ \\S+ \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | parse body /"statusCode":(?<sc>\\d+)/'
        ' | filter sc >= 400'
        ' | stats count(*) as errors by bin(1h) as hour'
        ' | sort hour asc',
        hours=insight_hours,
    )

    # --- Recent failures with mint addresses ---
    swap_recent = query_render_logs_stats(
        'filter @message like /swap-server/ and @message like /http-request/'
        ' | parse @message /^<\\d+>\\d+ \\S+ \\S+ \\S+ \\d+ \\S+ - (?<body>.*)/'
        ' | parse body /"statusCode":(?<sc>\\d+)/'
        ' | filter sc >= 400'
        ' | parse body /"time":"(?<ts>[^"]+)"/'
        ' | parse body /"method":"(?<method>[^"]+)"/'
        ' | parse body /"path":"(?<rawpath>[^"]+)"/'
        ' | parse body /"responseTimeMS":(?<latency>\\d+)/'
        ' | parse rawpath /^\\/(?<endpoint>[^?]+)/'
        ' | parse rawpath /inputMint=(?<in_mint>[^&]+)/'
        ' | parse rawpath /outputMint=(?<out_mint>[^&]+)/'
        ' | fields ts, sc, method, endpoint, latency, in_mint, out_mint'
        ' | sort @timestamp desc'
        ' | limit 25',
        hours=insight_hours,
    )

    col_swap_chart, col_swap_breakdown = st.columns(2)
    with col_swap_chart:
        if not swap_errors_timeline.empty and "errors" in swap_errors_timeline.columns:
            swap_errors_timeline["errors"] = pd.to_numeric(swap_errors_timeline["errors"], errors="coerce")
            if "hour" in swap_errors_timeline.columns:
                swap_errors_timeline["hour"] = pd.to_datetime(swap_errors_timeline["hour"], errors="coerce")
            fig = go.Figure(go.Bar(
                x=swap_errors_timeline.get("hour", swap_errors_timeline.index),
                y=swap_errors_timeline["errors"],
                marker_color=ACCENT_WARN,
                hovertemplate="<b>%{x}</b><br>%{y} failed requests<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Swap Server Failed Requests per Hour", height=280)
            fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
            fig.update_yaxes(title="count", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No swap server errors")

    with col_swap_breakdown:
        if not swap_errors.empty and "cnt" in swap_errors.columns:
            swap_errors["cnt"] = pd.to_numeric(swap_errors["cnt"], errors="coerce")
            labels = []
            for _, r in swap_errors.iterrows():
                sc = r.get("sc", "?")
                ep = r.get("endpoint", "?")
                mint = r.get("out_mint", "")
                mint_short = f" → {mint[:6]}...{mint[-4:]}" if isinstance(mint, str) and len(mint) > 10 else ""
                labels.append(f"HTTP {sc} /{ep}{mint_short}")
            fig = go.Figure(go.Bar(
                y=labels, x=swap_errors["cnt"],
                orientation="h", marker_color=ACCENT_WARN,
                text=[str(int(v)) for v in swap_errors["cnt"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>%{x} occurrences<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Failures by Status + Endpoint", height=280,
                              yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(title="", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    if not swap_recent.empty:
        display_cols = [c for c in ["ts", "sc", "method", "endpoint", "latency", "in_mint", "out_mint"] if c in swap_recent.columns]
        if display_cols:
            swap_display = swap_recent[display_cols].copy()
            # Shorten mint addresses for readability
            for col in ["in_mint", "out_mint"]:
                if col in swap_display.columns:
                    swap_display[col] = swap_display[col].apply(
                        lambda x: f"{x[:6]}...{x[-4:]}" if isinstance(x, str) and len(x) > 12 else x
                    )
            swap_display = swap_display.rename(columns={
                "ts": "Time", "sc": "Status", "method": "Method",
                "endpoint": "Endpoint", "latency": "Latency (ms)",
                "in_mint": "Input Token", "out_mint": "Output Token",
            })
            st.dataframe(swap_display, use_container_width=True, hide_index=True)

    # --- Twitter scraper: rejection stats ---
    st.markdown("---")
    st.markdown("### Twitter Scraper — Signal Analysis")

    scraper_stats = query_render_logs_stats(
        'filter @message like /twitter-scraper/'
        ' | parse @message />>> (?<outcome>\\S+)/'
        ' | filter outcome in ["SIGNAL:", "Not", "Written"]'
        ' | stats count(*) as cnt by outcome'
        ' | sort cnt desc',
        hours=insight_hours,
    )

    scraper_signals = query_render_logs_stats(
        'filter @message like /twitter-scraper/ and @message like /SIGNAL:/'
        ' | parse @message />>> SIGNAL: (?<ticker>\\S+) (?<direction>\\S+)/'
        ' | stats count(*) as cnt by ticker, direction'
        ' | sort cnt desc'
        ' | limit 15',
        hours=insight_hours,
    )

    col_scrape_pie, col_scrape_signals = st.columns(2)
    with col_scrape_pie:
        if not scraper_stats.empty and "cnt" in scraper_stats.columns and "outcome" in scraper_stats.columns:
            scraper_stats["cnt"] = pd.to_numeric(scraper_stats["cnt"], errors="coerce")
            # Map outcome labels
            label_map = {"SIGNAL:": "Signals Generated", "Not": "Rejected (Not Tradable)", "Written": "Written to DB/Sheet"}
            scraper_stats["label"] = scraper_stats["outcome"].map(label_map).fillna(scraper_stats["outcome"])
            color_map = {"Signals Generated": ACCENT, "Rejected (Not Tradable)": ACCENT_WARN, "Written to DB/Sheet": BRAND}
            colors = [color_map.get(l, TEXT_MUTED) for l in scraper_stats["label"]]
            fig = go.Figure(go.Pie(
                labels=scraper_stats["label"], values=scraper_stats["cnt"],
                marker=dict(colors=colors),
                textinfo="label+value", hole=0.4,
                hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Tweet Analysis Outcomes", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scraper data available")

    with col_scrape_signals:
        if not scraper_signals.empty and "cnt" in scraper_signals.columns and "ticker" in scraper_signals.columns:
            scraper_signals["cnt"] = pd.to_numeric(scraper_signals["cnt"], errors="coerce")
            direction = scraper_signals.get("direction", pd.Series([""] * len(scraper_signals)))
            colors = [ACCENT if d == "BUY" else ACCENT_RED if d == "SELL" else TEXT_MUTED for d in direction]
            labels = [f"{r.get('ticker', '?')} {r.get('direction', '')}" for _, r in scraper_signals.iterrows()]
            fig = go.Figure(go.Bar(
                y=labels, x=scraper_signals["cnt"],
                orientation="h", marker_color=colors,
                text=[str(int(v)) for v in scraper_signals["cnt"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>%{x} signals<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Trade Signals Generated", height=300,
                              yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
            fig.update_xaxes(title="", **AXIS_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # SECTION 4: CloudFront CDN
    # ============================================================
    st.markdown("---")
    st.markdown("### CloudFront CDN")
    st.caption("Trade feed CDN — serves /trades and /top-trades to all app clients")

    cf_dist = get_cloudfront_dist_id()
    if cf_dist:
        cf_requests = load_cloudfront_metric("Requests", stat="Sum", period=sh_period, hours=sh_hours, dist_id=cf_dist)
        cf_bytes = load_cloudfront_metric("BytesDownloaded", stat="Sum", period=sh_period, hours=sh_hours, dist_id=cf_dist)
        cf_4xx = load_cloudfront_metric("4xxErrorRate", stat="Average", period=sh_period, hours=sh_hours, dist_id=cf_dist)
        cf_5xx = load_cloudfront_metric("5xxErrorRate", stat="Average", period=sh_period, hours=sh_hours, dist_id=cf_dist)
        cf_total_hit = load_cloudfront_metric("CacheHitRate", stat="Average", period=sh_period, hours=sh_hours, dist_id=cf_dist)

        has_cf = not cf_requests.empty or not cf_bytes.empty

        if has_cf:
            # Summary metrics
            total_cf_reqs = int(cf_requests["value"].sum()) if not cf_requests.empty else 0
            total_cf_bytes_gb = cf_bytes["value"].sum() / (1024**3) if not cf_bytes.empty else 0
            avg_4xx = cf_4xx["value"].mean() if not cf_4xx.empty else 0
            avg_5xx = cf_5xx["value"].mean() if not cf_5xx.empty else 0
            avg_cache_hit = cf_total_hit["value"].mean() if not cf_total_hit.empty else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Requests", fmt_number(total_cf_reqs))
            c2.metric("Data Served", f"{total_cf_bytes_gb:.2f} GB")
            c3.metric("Cache Hit Rate", f"{avg_cache_hit:.1f}%")
            c4.metric("4xx Error Rate", f"{avg_4xx:.2f}%",
                       delta=f"{avg_4xx:.2f}%" if avg_4xx > 1 else None, delta_color="inverse")
            c5.metric("5xx Error Rate", f"{avg_5xx:.2f}%",
                       delta=f"{avg_5xx:.2f}%" if avg_5xx > 0.5 else None, delta_color="inverse")

            # Charts side by side
            col_cf_req, col_cf_bytes = st.columns(2)
            with col_cf_req:
                if not cf_requests.empty:
                    fig = make_time_series(cf_requests, "timestamp", "value",
                                           title="CDN Requests", color=BRAND, kind="bar")
                    st.plotly_chart(fig, use_container_width=True)
            with col_cf_bytes:
                if not cf_bytes.empty:
                    cf_bytes_mb = cf_bytes.copy()
                    cf_bytes_mb["value"] = cf_bytes_mb["value"] / (1024 * 1024)
                    fig = make_time_series(cf_bytes_mb, "timestamp", "value",
                                           title="Data Downloaded (MB)", color=ACCENT, kind="area")
                    st.plotly_chart(fig, use_container_width=True)

            col_cf_err, col_cf_cache = st.columns(2)
            with col_cf_err:
                # Combined error rate chart
                err_frames = []
                if not cf_4xx.empty:
                    e4 = cf_4xx.copy()
                    e4["type"] = "4xx"
                    err_frames.append(e4)
                if not cf_5xx.empty:
                    e5 = cf_5xx.copy()
                    e5["type"] = "5xx"
                    err_frames.append(e5)
                if err_frames:
                    edf = pd.concat(err_frames, ignore_index=True)
                    fig = px.line(edf, x="timestamp", y="value", color="type",
                                  title="CDN Error Rates (%)", color_discrete_sequence=[ACCENT_WARN, ACCENT_RED])
                    fig.update_layout(**PLOTLY_LAYOUT, height=300, legend=dict(orientation="h", y=-0.25))
                    fig.update_xaxes(title="", tickformat="%H:%M", **AXIS_DEFAULTS)
                    fig.update_yaxes(title="%", **AXIS_DEFAULTS)
                    st.plotly_chart(fig, use_container_width=True)

            with col_cf_cache:
                if not cf_total_hit.empty:
                    fig = make_time_series(cf_total_hit, "timestamp", "value",
                                           title="Cache Hit Rate (%)", color="#06B6D4", kind="area")
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CloudFront data available for this period.")
    else:
        st.info("No CloudFront distributions found in CloudWatch. CDN metrics will appear once the distribution is active and receiving traffic.")

    # ============================================================
    # SECTION 5: Log Viewer
    # ============================================================
    st.markdown("---")
    st.markdown("### Log Viewer")
    st.caption("Syslog from Render → OTel Collector → `/render/logs`")

    col_lf1, col_lf2, col_lf3 = st.columns(3)
    with col_lf1:
        log_svc = st.selectbox("Service", ["all"] + RENDER_SERVICES, key="log_svc")
    with col_lf2:
        log_level = st.selectbox("Level", ["all", "errors only", "warnings+"], key="log_level")
    with col_lf3:
        log_hours = st.selectbox("Log lookback", [1, 6, 12, 24], index=0, key="log_hours",
                                  format_func=lambda h: f"{h}h")

    log_search = st.text_input("Search (optional)", placeholder="error, timeout, 500...", key="log_search")

    if st.button("Fetch Logs", key="fetch_logs"):
        filters = []
        if log_svc != "all":
            filters.append(f'filter @message like /{log_svc}/')
        if log_level == "errors only":
            filters.append('filter @message like /(?i)(error|exception|fatal|panic|crash|ECONNREFUSED|ETIMEDOUT)/')
        elif log_level == "warnings+":
            filters.append('filter @message like /(?i)(error|warn|exception|fatal|panic|crash|timeout|fail)/')
        if log_search:
            filters.append(f'filter @message like /{log_search}/')

        filter_clause = " | ".join(filters)
        query = f'fields @timestamp, @message | {filter_clause + " | " if filter_clause else ""}sort @timestamp desc | limit 200'

        with st.spinner("Querying CloudWatch Logs..."):
            log_df = query_render_logs("/render/logs", query, hours=log_hours, limit=200)

        if log_df.empty:
            st.info("No logs found for the selected filters.")
        else:
            st.caption(f"Showing {len(log_df)} log entries")
            st.dataframe(log_df, use_container_width=True, height=500)
