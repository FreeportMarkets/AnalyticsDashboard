import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    # Opens: collateral × leverage = notional
    if "leverage" in frame.columns:
        frame["leverage"] = pd.to_numeric(frame["leverage"], errors="coerce").fillna(1)
        frame.loc[open_mask, "amount_usd"] = frame.loc[open_mask, "amount_usd"] * frame.loc[open_mask, "leverage"]
    # Closes: size × price = full position notional
    if "size" in frame.columns and "price" in frame.columns:
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce").fillna(0)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0)
        frame.loc[close_mask, "amount_usd"] = frame.loc[close_mask, "size"].abs() * frame.loc[close_mask, "price"]
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
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=st.secrets["aws"]["region"],
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    )


ANALYTICS_TABLE = "freeport-analytics-events"
TRADES_TABLE = "freeport-trades-history"


# --- Data Loading ---
@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def load_events_range(start_date: str, end_date: str) -> list:
    all_items = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        all_items.extend(load_events_for_date(current.strftime("%Y-%m-%d")))
        current += timedelta(days=1)
    return all_items


@st.cache_data(ttl=300)
def load_trades_range(start_date: str, end_date: str) -> list:
    db = get_dynamodb()
    table = db.Table(TRADES_TABLE)
    items, params = [], {
        "FilterExpression": Key("timestamp").between(start_date + "T00:00:00", end_date + "T23:59:59"),
    }
    while True:
        resp = table.scan(**params)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        params["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return [decimal_to_float(i) for i in items]


def events_to_df(events: list) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df["ts"] = ts.dt.tz_convert("America/New_York")
        # Re-derive hour and day_of_week in EST
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

today = datetime.utcnow().date()
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

with st.spinner("Loading analytics data..."):
    events = load_events_range(start_str, end_str)
    df = events_to_df(events)

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
tab_overview, tab_users, tab_retention, tab_funnels, tab_trades, tab_notifications = st.tabs(
    ["Overview", "Top Users", "Retention", "Funnels", "Trades", "Notifications"]
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
        with st.spinner("Loading trade data..."):
            trades = load_trades_range(start_str, end_str)

        if trades:
            tdf = pd.DataFrame(trades)
            # Exclude deposits from trader leaderboards
            if "type" in tdf.columns:
                tdf = tdf[tdf["type"] != "deposit"]
            if not tdf.empty and "amount_usd" in tdf.columns and "wallet_address" in tdf.columns:
                tdf["amount_usd"] = pd.to_numeric(tdf["amount_usd"], errors="coerce")
                tdf = apply_perps_leverage(tdf)

                col_vol, col_count = st.columns(2)

                with col_vol:
                    trader_vol = tdf.groupby("wallet_address").agg(
                        volume=("amount_usd", "sum"), trades=("amount_usd", "count"),
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

            if trades:
                user_trades = [t for t in trades if t.get("wallet_address") == full_wallet]
                if user_trades:
                    utdf = pd.DataFrame(user_trades)
                    if "amount_usd" in utdf.columns:
                        utdf["amount_usd"] = pd.to_numeric(utdf["amount_usd"], errors="coerce")
                        utdf = apply_perps_leverage(utdf)
                        st.markdown(f"**Trades:** {len(utdf)} trades | **${utdf['amount_usd'].sum():,.2f}** total volume")
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
    with st.spinner("Loading trade data..."):
        trades = load_trades_range(start_str, end_str)

    if not trades:
        st.info("No trades found for this date range.")
    else:
        tdf = pd.DataFrame(trades)
        if "amount_usd" in tdf.columns:
            tdf["amount_usd"] = pd.to_numeric(tdf["amount_usd"], errors="coerce")
        if "timestamp" in tdf.columns:
            ts = pd.to_datetime(tdf["timestamp"], errors="coerce")
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            tdf["ts"] = ts.dt.tz_convert("America/New_York")
            tdf["trade_date"] = tdf["ts"].dt.strftime("%Y-%m-%d")
            tdf["hour"] = tdf["ts"].dt.hour
            tdf["dow"] = tdf["ts"].dt.dayofweek

        # Split by type
        swap_df = tdf[tdf["type"] == "swap"].copy() if "type" in tdf.columns else tdf.copy()
        perps_df = tdf[tdf["type"] == "perps"].copy() if "type" in tdf.columns else pd.DataFrame()
        deposit_df = tdf[tdf["type"] == "deposit"].copy() if "type" in tdf.columns else pd.DataFrame()

        # Compute notional volume: opens = amount*leverage, closes = size*price
        if not perps_df.empty and "amount_usd" in perps_df.columns:
            if "leverage" in perps_df.columns:
                perps_df["leverage"] = pd.to_numeric(perps_df["leverage"], errors="coerce").fillna(1)
            if "size" in perps_df.columns:
                perps_df["size"] = pd.to_numeric(perps_df["size"], errors="coerce").fillna(0)
            if "price" in perps_df.columns:
                perps_df["price"] = pd.to_numeric(perps_df["price"], errors="coerce").fillna(0)

            close_mask = perps_df["is_close"].astype(bool) if "is_close" in perps_df.columns else pd.Series(False, index=perps_df.index)
            open_mask = ~close_mask

            # Opens: collateral × leverage
            if "leverage" in perps_df.columns:
                perps_df.loc[open_mask, "amount_usd"] = perps_df.loc[open_mask, "amount_usd"] * perps_df.loc[open_mask, "leverage"]
            # Closes: size × price (full position notional)
            if "size" in perps_df.columns and "price" in perps_df.columns:
                perps_df.loc[close_mask, "amount_usd"] = perps_df.loc[close_mask, "size"].abs() * perps_df.loc[close_mask, "price"]

        trades_only = pd.concat([swap_df, perps_df], ignore_index=True)  # exclude deposits

        # --- Top-level metrics (swaps + perps only) ---
        total_vol = trades_only["amount_usd"].sum() if "amount_usd" in trades_only.columns and not trades_only.empty else 0
        total_trades = len(trades_only)
        unique_traders = trades_only["wallet_address"].nunique() if "wallet_address" in trades_only.columns and not trades_only.empty else 0
        avg_trade = total_vol / total_trades if total_trades > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trading Volume", f"${fmt_number(total_vol)}")
        col2.metric("Total Trades", fmt_number(total_trades))
        col3.metric("Unique Traders", fmt_number(unique_traders))
        col4.metric("Avg Trade Size", f"${avg_trade:,.0f}")

        # Type breakdown row
        swap_vol = swap_df["amount_usd"].sum() if not swap_df.empty and "amount_usd" in swap_df.columns else 0
        perps_vol = perps_df["amount_usd"].sum() if not perps_df.empty and "amount_usd" in perps_df.columns else 0
        deposit_vol = deposit_df["amount_usd"].sum() if not deposit_df.empty and "amount_usd" in deposit_df.columns else 0
        cb1, cb2, cb3 = st.columns(3)
        cb1.metric("Swap Volume", f"${fmt_number(swap_vol)}", delta=f"{len(swap_df)} swaps")
        cb2.metric("Perps Volume", f"${fmt_number(perps_vol)}", delta=f"{len(perps_df)} orders")
        cb3.metric("Deposit Volume", f"${fmt_number(deposit_vol)}", delta=f"{len(deposit_df)} deposits")

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
            perps_total_vol = perps_df["amount_usd"].sum() if "amount_usd" in perps_df.columns else 0
            perps_total_orders = len(perps_df)
            perps_unique = perps_df["wallet_address"].nunique() if "wallet_address" in perps_df.columns else 0
            perps_avg = perps_total_vol / perps_total_orders if perps_total_orders > 0 else 0

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Perps Volume", f"${fmt_number(perps_total_vol)}")
            pc2.metric("Total Orders", fmt_number(perps_total_orders))
            pc3.metric("Unique Perps Traders", fmt_number(perps_unique))
            pc4.metric("Avg Order Size", f"${perps_avg:,.0f}")

            # Daily perps volume
            if "trade_date" in perps_df.columns and "amount_usd" in perps_df.columns:
                daily_perps = perps_df.groupby("trade_date")["amount_usd"].sum().reset_index()
                daily_perps.columns = ["date", "volume"]
                fig = make_time_series(daily_perps, "date", "volume", title="Daily Perps Volume ($)", color="#8B5CF6", kind="bar")
                st.plotly_chart(fig, use_container_width=True)

            # Volume by Asset + Orders by Asset
            col_pv, col_po = st.columns(2)

            with col_pv:
                if "asset" in perps_df.columns and "amount_usd" in perps_df.columns:
                    pva = perps_df.groupby("asset")["amount_usd"].sum().sort_values(ascending=False).head(15).reset_index()
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
                perps_df["order_side"] = perps_df["is_close"].map(lambda x: "Close" if x else "Open")
                oc_agg = perps_df.groupby("order_side").agg(
                    volume=("amount_usd", "sum"), count=("amount_usd", "size"),
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
