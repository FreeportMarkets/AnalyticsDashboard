import streamlit as st
import boto3
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Freeport Analytics", page_icon="📊", layout="wide")

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
@st.cache_data(ttl=300)  # 5-minute cache
def load_events_for_date(date_str: str) -> list:
    db = get_dynamodb()
    table = db.Table(ANALYTICS_TABLE)
    items = []
    params = {
        "KeyConditionExpression": Key("date").eq(date_str),
    }
    while True:
        resp = table.query(**params)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        params["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


@st.cache_data(ttl=300)
def load_events_range(start_date: str, end_date: str) -> list:
    """Load events across a date range by querying each day."""
    all_items = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        items = load_events_for_date(current.strftime("%Y-%m-%d"))
        all_items.extend(items)
        current += timedelta(days=1)
    return all_items


@st.cache_data(ttl=300)
def load_trades_range(start_date: str, end_date: str) -> list:
    """Scan trades table filtered by date range."""
    db = get_dynamodb()
    table = db.Table(TRADES_TABLE)
    items = []
    params = {
        "FilterExpression": Key("timestamp").between(
            start_date + "T00:00:00", end_date + "T23:59:59"
        ),
    }
    while True:
        resp = table.scan(**params)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        params["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def events_to_df(events: list) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    if "day_of_week" in df.columns:
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce")
    return df


# --- Sidebar ---
st.sidebar.title("Freeport Analytics")

# Date range selector
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

# Load data
with st.spinner("Loading analytics data..."):
    events = load_events_range(start_str, end_str)
    df = events_to_df(events)

st.sidebar.metric("Total Events", f"{len(df):,}")

# --- Tabs ---
tab_overview, tab_retention, tab_funnels, tab_trades, tab_notifications = st.tabs(
    ["Overview", "Retention", "Funnels", "Trades", "Notifications"]
)


# =====================
# TAB 1: Overview
# =====================
with tab_overview:
    st.header("Overview")

    if df.empty:
        st.info("No analytics events found for this date range.")
    else:
        # Unique users per day
        daily_users = df.groupby("date")["wallet_address"].nunique()
        total_unique = df["wallet_address"].nunique()

        # Session counts
        sessions = df[df["event"] == "session_start"]
        session_ends = df[df["event"] == "session_end"]

        # Avg session duration
        avg_duration_ms = 0
        if not session_ends.empty and "metadata" in session_ends.columns:
            durations = []
            for _, row in session_ends.iterrows():
                meta = row.get("metadata")
                if isinstance(meta, dict) and "duration_ms" in meta:
                    d = meta["duration_ms"]
                    if isinstance(d, (int, float)) and d > 0:
                        durations.append(d)
            if durations:
                avg_duration_ms = sum(durations) / len(durations)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DAU (avg)", f"{daily_users.mean():.0f}" if not daily_users.empty else "0")
        col2.metric("Unique Users", f"{total_unique:,}")
        col3.metric("Total Sessions", f"{len(sessions):,}")
        col4.metric("Avg Session", f"{avg_duration_ms / 1000:.0f}s" if avg_duration_ms else "N/A")

        # WAU/MAU
        if len(daily_users) >= 7:
            wau = df[df["date"] >= (today - timedelta(days=7)).strftime("%Y-%m-%d")]["wallet_address"].nunique()
            st.metric("WAU (last 7 days)", f"{wau:,}")

        # Events by day chart
        st.subheader("Events by Day")
        events_by_day = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(events_by_day.set_index("date")["count"])

        # Top events
        st.subheader("Top Events")
        top_events = df["event"].value_counts().head(15)
        st.dataframe(top_events.reset_index().rename(columns={"index": "Event", "event": "Event", "count": "Count"}))

        # Hourly activity
        if "hour" in df.columns:
            st.subheader("Hourly Activity (UTC)")
            hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
            st.bar_chart(hourly)


# =====================
# TAB 2: Retention
# =====================
with tab_retention:
    st.header("Cohort Retention")

    if df.empty:
        st.info("No data for retention analysis.")
    else:
        # Build cohorts: group users by first-seen date
        first_seen = df.groupby("wallet_address")["date"].min().reset_index()
        first_seen.columns = ["wallet_address", "cohort_date"]
        merged = df.merge(first_seen, on="wallet_address")

        cohort_dates = sorted(first_seen["cohort_date"].unique())
        retention_days = [0, 1, 3, 7, 14, 30]

        retention_data = []
        for cohort in cohort_dates:
            cohort_users = set(first_seen[first_seen["cohort_date"] == cohort]["wallet_address"])
            cohort_size = len(cohort_users)
            if cohort_size == 0:
                continue

            row = {"Cohort": cohort, "Users": cohort_size}
            cohort_dt = datetime.strptime(cohort, "%Y-%m-%d")

            for d in retention_days:
                target_date = (cohort_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                active_on_day = set(
                    df[df["date"] == target_date]["wallet_address"]
                )
                retained = cohort_users & active_on_day
                pct = (len(retained) / cohort_size * 100) if cohort_size > 0 else 0
                row[f"D{d}"] = f"{pct:.0f}%"

            retention_data.append(row)

        if retention_data:
            st.dataframe(pd.DataFrame(retention_data), use_container_width=True)
        else:
            st.info("Not enough data for cohort analysis.")


# =====================
# TAB 3: Funnels
# =====================
with tab_funnels:
    st.header("Funnels")

    if df.empty:
        st.info("No data for funnel analysis.")
    else:
        # Notification funnel
        st.subheader("Notification → Trade Funnel")
        notif_received = df[df["event"] == "notification_received"]["wallet_address"].nunique()
        notif_tapped = df[df["event"] == "notification_tap"]["wallet_address"].nunique()
        trade_init = df[df["event"] == "trade_initiated"]["wallet_address"].nunique()
        trade_success = df[df["event"] == "trade_success"]["wallet_address"].nunique()

        funnel_data = pd.DataFrame({
            "Step": ["Notification Received", "Notification Tapped", "Trade Initiated", "Trade Success"],
            "Users": [notif_received, notif_tapped, trade_init, trade_success],
        })
        st.dataframe(funnel_data, use_container_width=True)

        # Deposit funnel
        st.subheader("Deposit Funnel")
        dep_init = df[df["event"] == "deposit_initiated"]["wallet_address"].nunique()
        dep_success = df[df["event"] == "deposit_success"]["wallet_address"].nunique()
        dep_error = df[df["event"] == "deposit_error"]["wallet_address"].nunique()

        dep_funnel = pd.DataFrame({
            "Step": ["Deposit Initiated", "Deposit Success", "Deposit Error"],
            "Users": [dep_init, dep_success, dep_error],
        })
        st.dataframe(dep_funnel, use_container_width=True)

        # Trade funnel (detail)
        st.subheader("Trade Funnel")
        swap_modal = df[df["event"] == "swap_modal_open"]["wallet_address"].nunique()
        t_init = df[df["event"] == "trade_initiated"]["wallet_address"].nunique()
        t_success = df[df["event"] == "trade_success"]["wallet_address"].nunique()
        t_error = df[df["event"] == "trade_error"]["wallet_address"].nunique()

        trade_funnel = pd.DataFrame({
            "Step": ["Swap Modal Open", "Trade Initiated", "Trade Success", "Trade Error"],
            "Users": [swap_modal, t_init, t_success, t_error],
        })
        st.dataframe(trade_funnel, use_container_width=True)


# =====================
# TAB 4: Trades
# =====================
with tab_trades:
    st.header("Trades")

    with st.spinner("Loading trade data..."):
        trades = load_trades_range(start_str, end_str)

    if not trades:
        st.info("No trades found for this date range.")
    else:
        tdf = pd.DataFrame(trades)
        if "amount_usd" in tdf.columns:
            tdf["amount_usd"] = pd.to_numeric(tdf["amount_usd"], errors="coerce")
        if "timestamp" in tdf.columns:
            tdf["ts"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
            tdf["date"] = tdf["ts"].dt.strftime("%Y-%m-%d")
            tdf["hour"] = tdf["ts"].dt.hour
            tdf["dow"] = tdf["ts"].dt.dayofweek

        total_vol = tdf["amount_usd"].sum() if "amount_usd" in tdf.columns else 0
        total_trades = len(tdf)

        col1, col2 = st.columns(2)
        col1.metric("Total Volume", f"${total_vol:,.0f}")
        col2.metric("Total Trades", f"{total_trades:,}")

        # Volume by token
        if "to_token" in tdf.columns and "amount_usd" in tdf.columns:
            st.subheader("Volume by Token")
            vol_by_token = tdf.groupby("to_token")["amount_usd"].sum().sort_values(ascending=False).head(20)
            st.bar_chart(vol_by_token)

        # Trades by hour
        if "hour" in tdf.columns:
            st.subheader("Trades by Hour (UTC)")
            by_hour = tdf.groupby("hour").size().reindex(range(24), fill_value=0)
            st.bar_chart(by_hour)

        # Trades by day of week
        if "dow" in tdf.columns:
            st.subheader("Trades by Day of Week")
            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            by_dow = tdf.groupby("dow").size().reindex(range(7), fill_value=0)
            by_dow.index = dow_labels
            st.bar_chart(by_dow)


# =====================
# TAB 5: Notifications
# =====================
with tab_notifications:
    st.header("Notifications")

    if df.empty:
        st.info("No data for notification analysis.")
    else:
        received_events = df[df["event"] == "notification_received"]
        tapped_events = df[df["event"] == "notification_tap"]

        received_count = len(received_events)
        tapped_count = len(tapped_events)
        tap_rate = (tapped_count / received_count * 100) if received_count > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Received", f"{received_count:,}")
        col2.metric("Tapped", f"{tapped_count:,}")
        col3.metric("Tap Rate", f"{tap_rate:.1f}%")

        # Time to trade after notification tap
        st.subheader("Time to Trade After Notification Tap")
        if not tapped_events.empty and "ts" in df.columns:
            trade_events = df[df["event"] == "trade_initiated"].copy()
            if not trade_events.empty:
                times_to_trade = []
                for _, tap in tapped_events.iterrows():
                    wallet = tap["wallet_address"]
                    tap_ts = tap.get("ts")
                    if pd.isna(tap_ts):
                        continue
                    # Find next trade by same wallet within 1 hour
                    wallet_trades = trade_events[
                        (trade_events["wallet_address"] == wallet)
                        & (trade_events["ts"] > tap_ts)
                        & (trade_events["ts"] < tap_ts + timedelta(hours=1))
                    ]
                    if not wallet_trades.empty:
                        first_trade = wallet_trades.iloc[0]
                        delta = (first_trade["ts"] - tap_ts).total_seconds()
                        times_to_trade.append(delta)

                if times_to_trade:
                    avg_seconds = sum(times_to_trade) / len(times_to_trade)
                    median_seconds = sorted(times_to_trade)[len(times_to_trade) // 2]
                    st.metric("Avg Time to Trade", f"{avg_seconds:.0f}s")
                    st.metric("Median Time to Trade", f"{median_seconds:.0f}s")
                    st.metric("Conversions (tap→trade <1hr)", f"{len(times_to_trade)}")
                else:
                    st.info("No notification→trade conversions found within 1 hour.")
            else:
                st.info("No trade events found.")
        else:
            st.info("No notification tap events found.")

        # Top tapped tickers
        if not tapped_events.empty and "metadata" in tapped_events.columns:
            st.subheader("Most Tapped Tickers")
            tickers = []
            for _, row in tapped_events.iterrows():
                meta = row.get("metadata")
                if isinstance(meta, dict) and meta.get("ticker"):
                    tickers.append(meta["ticker"])
            if tickers:
                ticker_counts = pd.Series(tickers).value_counts().head(10)
                st.bar_chart(ticker_counts)
