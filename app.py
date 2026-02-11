import streamlit as st
import boto3
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from boto3.dynamodb.conditions import Key

st.set_page_config(page_title="Freeport Analytics", page_icon="📊", layout="wide")


# --- Helpers ---
def short_wallet(w):
    if not w or len(w) < 8:
        return w or "?"
    return f"{w[:4]}...{w[-4:]}"


def decimal_to_float(obj):
    """Convert DynamoDB Decimal types to float for pandas."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    return obj


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
    items = []
    params = {"KeyConditionExpression": Key("date").eq(date_str)}
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
    return [decimal_to_float(i) for i in items]


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


def extract_session_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-session duration from session_end events."""
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
st.sidebar.title("Freeport Analytics")

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

with st.spinner("Loading analytics data..."):
    events = load_events_range(start_str, end_str)
    df = events_to_df(events)

st.sidebar.metric("Total Events", f"{len(df):,}")
if not df.empty:
    st.sidebar.metric("Unique Users", f"{df['wallet_address'].nunique():,}")

# --- Tabs ---
tab_overview, tab_users, tab_retention, tab_funnels, tab_trades, tab_notifications = st.tabs(
    ["Overview", "Top Users", "Retention", "Funnels", "Trades", "Notifications"]
)


# =====================
# TAB 1: Overview
# =====================
with tab_overview:
    st.header("Overview")

    if df.empty:
        st.info("No analytics events found for this date range.")
    else:
        daily_users = df.groupby("date")["wallet_address"].nunique()
        total_unique = df["wallet_address"].nunique()

        sessions = df[df["event"] == "session_start"]
        sess_durations = extract_session_durations(df)
        avg_duration_ms = sess_durations["duration_ms"].mean() if not sess_durations.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DAU (avg)", f"{daily_users.mean():.0f}" if not daily_users.empty else "0")
        col2.metric("Unique Users", f"{total_unique:,}")
        col3.metric("Total Sessions", f"{len(sessions):,}")
        col4.metric("Avg Session", f"{avg_duration_ms / 1000:.0f}s" if avg_duration_ms else "N/A")

        # DAU over time
        st.subheader("Daily Active Users")
        st.line_chart(daily_users)

        # WAU/MAU
        col_w, col_m = st.columns(2)
        wau = df[df["date"] >= (today - timedelta(days=7)).strftime("%Y-%m-%d")]["wallet_address"].nunique()
        mau = df[df["date"] >= (today - timedelta(days=30)).strftime("%Y-%m-%d")]["wallet_address"].nunique()
        col_w.metric("WAU (last 7 days)", f"{wau:,}")
        col_m.metric("MAU (last 30 days)", f"{mau:,}")

        # Events by day
        st.subheader("Events by Day")
        events_by_day = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(events_by_day.set_index("date")["count"])

        # Top events
        st.subheader("Top Events")
        top_events = df["event"].value_counts().head(15).reset_index()
        top_events.columns = ["Event", "Count"]
        st.dataframe(top_events, use_container_width=True)

        # Hourly activity
        if "hour" in df.columns:
            st.subheader("Hourly Activity (UTC)")
            hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
            st.bar_chart(hourly)

        # Daily activity by day of week
        if "day_of_week" in df.columns:
            st.subheader("Activity by Day of Week")
            dow_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            by_dow = df.groupby("day_of_week").size().reindex(range(7), fill_value=0)
            by_dow.index = dow_labels
            st.bar_chart(by_dow)


# =====================
# TAB 2: Top Users
# =====================
with tab_users:
    st.header("Top Users")

    if df.empty:
        st.info("No data available.")
    else:
        # --- Time Spent Leaderboard ---
        st.subheader("Most Time Spent in App")
        sess_durations = extract_session_durations(df)
        if not sess_durations.empty:
            time_by_user = sess_durations.groupby("wallet_address").agg(
                total_time_min=("duration_ms", lambda x: round(x.sum() / 60000, 1)),
                sessions=("session_id", "count"),
                avg_session_min=("duration_ms", lambda x: round(x.mean() / 60000, 1)),
            ).sort_values("total_time_min", ascending=False).head(25).reset_index()
            time_by_user["wallet"] = time_by_user["wallet_address"].apply(short_wallet)
            st.dataframe(
                time_by_user[["wallet", "total_time_min", "sessions", "avg_session_min"]].rename(columns={
                    "wallet": "Wallet",
                    "total_time_min": "Total Time (min)",
                    "sessions": "Sessions",
                    "avg_session_min": "Avg Session (min)",
                }),
                use_container_width=True,
            )
        else:
            st.info("No session duration data yet.")

        # --- Most Active by Event Count ---
        st.subheader("Most Active Users (by event count)")
        events_per_user = df.groupby("wallet_address").agg(
            events=("event", "count"),
            days_active=("date", "nunique"),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
        ).sort_values("events", ascending=False).head(25).reset_index()
        events_per_user["wallet"] = events_per_user["wallet_address"].apply(short_wallet)
        st.dataframe(
            events_per_user[["wallet", "events", "days_active", "first_seen", "last_seen"]].rename(columns={
                "wallet": "Wallet",
                "events": "Events",
                "days_active": "Days Active",
                "first_seen": "First Seen",
                "last_seen": "Last Seen",
            }),
            use_container_width=True,
        )

        # --- Top Traders by Volume ---
        st.subheader("Top Traders (by volume)")
        with st.spinner("Loading trade data..."):
            trades = load_trades_range(start_str, end_str)

        if trades:
            tdf = pd.DataFrame(trades)
            if "amount_usd" in tdf.columns and "wallet_address" in tdf.columns:
                tdf["amount_usd"] = pd.to_numeric(tdf["amount_usd"], errors="coerce")
                trader_stats = tdf.groupby("wallet_address").agg(
                    volume=("amount_usd", "sum"),
                    trades=("amount_usd", "count"),
                    avg_trade=("amount_usd", "mean"),
                ).sort_values("volume", ascending=False).head(25).reset_index()
                trader_stats["wallet"] = trader_stats["wallet_address"].apply(short_wallet)
                trader_stats["volume"] = trader_stats["volume"].round(2)
                trader_stats["avg_trade"] = trader_stats["avg_trade"].round(2)
                st.dataframe(
                    trader_stats[["wallet", "volume", "trades", "avg_trade"]].rename(columns={
                        "wallet": "Wallet",
                        "volume": "Volume ($)",
                        "trades": "Trades",
                        "avg_trade": "Avg Trade ($)",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("Trade data missing expected fields.")
        else:
            st.info("No trade data for this period.")

        # --- Top Traders by Trade Count ---
        if trades:
            tdf = pd.DataFrame(trades)
            if "wallet_address" in tdf.columns:
                st.subheader("Most Frequent Traders (by count)")
                freq_traders = tdf.groupby("wallet_address").size().sort_values(ascending=False).head(25).reset_index()
                freq_traders.columns = ["wallet_address", "trade_count"]
                freq_traders["wallet"] = freq_traders["wallet_address"].apply(short_wallet)
                st.dataframe(
                    freq_traders[["wallet", "trade_count"]].rename(columns={
                        "wallet": "Wallet",
                        "trade_count": "Trades",
                    }),
                    use_container_width=True,
                )

        # --- User Activity Heatmap (hour × day of week) ---
        st.subheader("User Activity Heatmap (Hour × Day)")
        if "hour" in df.columns and "day_of_week" in df.columns:
            heatmap = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
            heatmap = heatmap.reindex(index=range(7), columns=range(24), fill_value=0)
            heatmap.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            heatmap.columns = [f"{h:02d}" for h in range(24)]
            try:
                st.dataframe(
                    heatmap.style.background_gradient(cmap="YlOrRd", axis=None),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(heatmap, use_container_width=True)

        # --- Per-User Deep Dive ---
        st.subheader("User Deep Dive")
        all_wallets = sorted(df["wallet_address"].unique())
        wallet_options = [f"{short_wallet(w)} ({w})" for w in all_wallets]
        selected = st.selectbox("Select a user", wallet_options, index=None, placeholder="Pick a wallet...")

        if selected:
            full_wallet = selected.split("(")[1].rstrip(")")
            user_df = df[df["wallet_address"] == full_wallet]

            col1, col2, col3 = st.columns(3)
            col1.metric("Events", f"{len(user_df):,}")
            col2.metric("Days Active", f"{user_df['date'].nunique()}")

            user_sessions = extract_session_durations(user_df)
            total_min = user_sessions["duration_ms"].sum() / 60000 if not user_sessions.empty else 0
            col3.metric("Total Time", f"{total_min:.1f} min")

            # User's event breakdown
            st.write("**Event breakdown:**")
            user_events = user_df["event"].value_counts().reset_index()
            user_events.columns = ["Event", "Count"]
            st.dataframe(user_events, use_container_width=True)

            # User's daily activity
            st.write("**Daily activity:**")
            user_daily = user_df.groupby("date").size().reset_index(name="events")
            st.bar_chart(user_daily.set_index("date")["events"])

            # User's trades
            if trades:
                user_trades = [t for t in trades if t.get("wallet_address") == full_wallet]
                if user_trades:
                    utdf = pd.DataFrame(user_trades)
                    if "amount_usd" in utdf.columns:
                        utdf["amount_usd"] = pd.to_numeric(utdf["amount_usd"], errors="coerce")
                        st.write(f"**Trades:** {len(utdf)} trades, ${utdf['amount_usd'].sum():,.2f} volume")
                        display_cols = [c for c in ["timestamp", "from_token", "to_token", "amount_usd", "status", "source"] if c in utdf.columns]
                        st.dataframe(utdf[display_cols].sort_values("timestamp", ascending=False), use_container_width=True)


# =====================
# TAB 3: Retention
# =====================
with tab_retention:
    st.header("Cohort Retention")

    if df.empty:
        st.info("No data for retention analysis.")
    else:
        first_seen = df.groupby("wallet_address")["date"].min().reset_index()
        first_seen.columns = ["wallet_address", "cohort_date"]

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
                active_on_day = set(df[df["date"] == target_date]["wallet_address"])
                retained = cohort_users & active_on_day
                pct = (len(retained) / cohort_size * 100) if cohort_size > 0 else 0
                row[f"D{d}"] = f"{pct:.0f}%"

            retention_data.append(row)

        if retention_data:
            ret_df = pd.DataFrame(retention_data)
            st.dataframe(ret_df, use_container_width=True)

            # Retention curve (average across cohorts)
            st.subheader("Average Retention Curve")
            avg_retention = {}
            for d in retention_days:
                col = f"D{d}"
                vals = [int(r[col].replace("%", "")) for r in retention_data if col in r]
                avg_retention[f"Day {d}"] = sum(vals) / len(vals) if vals else 0
            st.bar_chart(pd.Series(avg_retention))
        else:
            st.info("Not enough data for cohort analysis.")


# =====================
# TAB 4: Funnels
# =====================
with tab_funnels:
    st.header("Funnels")

    if df.empty:
        st.info("No data for funnel analysis.")
    else:
        # Notification → Trade funnel
        st.subheader("Notification → Trade Funnel")
        notif_received = df[df["event"] == "notification_received"]["wallet_address"].nunique()
        notif_tapped = df[df["event"] == "notification_tap"]["wallet_address"].nunique()
        trade_init = df[df["event"] == "trade_initiated"]["wallet_address"].nunique()
        trade_success = df[df["event"] == "trade_success"]["wallet_address"].nunique()

        funnel_data = pd.DataFrame({
            "Step": ["Notification Received", "Notification Tapped", "Trade Initiated", "Trade Success"],
            "Users": [notif_received, notif_tapped, trade_init, trade_success],
        })
        funnel_data["Conversion"] = ["—"] + [
            f"{(funnel_data['Users'].iloc[i] / funnel_data['Users'].iloc[i-1] * 100):.0f}%"
            if funnel_data["Users"].iloc[i-1] > 0 else "—"
            for i in range(1, len(funnel_data))
        ]
        st.dataframe(funnel_data, use_container_width=True)

        # Deposit funnel
        st.subheader("Deposit Funnel")
        dep_modal = df[df["event"] == "deposit_modal_open"]["wallet_address"].nunique()
        dep_init = df[df["event"] == "deposit_initiated"]["wallet_address"].nunique()
        dep_success = df[df["event"] == "deposit_success"]["wallet_address"].nunique()
        dep_error = df[df["event"] == "deposit_error"]["wallet_address"].nunique()

        dep_funnel = pd.DataFrame({
            "Step": ["Deposit Modal Open", "Deposit Initiated", "Deposit Success", "Deposit Error"],
            "Users": [dep_modal, dep_init, dep_success, dep_error],
        })
        st.dataframe(dep_funnel, use_container_width=True)

        # Deposit by provider
        st.subheader("Deposits by Provider")
        dep_events = df[df["event"].isin(["deposit_initiated", "deposit_success", "deposit_error"])]
        if not dep_events.empty and "metadata" in dep_events.columns:
            provider_rows = []
            for _, r in dep_events.iterrows():
                meta = r.get("metadata")
                if isinstance(meta, dict) and meta.get("provider"):
                    provider_rows.append({"provider": meta["provider"], "event": r["event"]})
            if provider_rows:
                prov_df = pd.DataFrame(provider_rows)
                prov_pivot = prov_df.groupby(["provider", "event"]).size().unstack(fill_value=0)
                st.dataframe(prov_pivot, use_container_width=True)

        # Trade funnel
        st.subheader("Trade Funnel")
        swap_modal = df[df["event"] == "swap_modal_open"]["wallet_address"].nunique()
        buy_tap = df[df["event"] == "buy_button_tap"]["wallet_address"].nunique()
        t_init = df[df["event"] == "trade_initiated"]["wallet_address"].nunique()
        t_success = df[df["event"] == "trade_success"]["wallet_address"].nunique()
        t_error = df[df["event"] == "trade_error"]["wallet_address"].nunique()

        trade_funnel = pd.DataFrame({
            "Step": ["Buy Button Tap", "Swap Modal Open", "Trade Initiated", "Trade Success", "Trade Error"],
            "Users": [buy_tap, swap_modal, t_init, t_success, t_error],
        })
        st.dataframe(trade_funnel, use_container_width=True)

        # Feature engagement
        st.subheader("Feature Engagement")
        engagement_events = [
            "feed_card_tap", "token_detail_view", "search_query", "category_select",
            "chart_period_change", "scroll_depth", "side_drawer_open",
            "perps_order_placed", "bridge_transfer", "app_open_deep_link",
        ]
        eng_data = []
        for evt in engagement_events:
            evt_df = df[df["event"] == evt]
            eng_data.append({
                "Event": evt,
                "Total": len(evt_df),
                "Unique Users": evt_df["wallet_address"].nunique(),
            })
        eng_df = pd.DataFrame(eng_data).sort_values("Total", ascending=False)
        st.dataframe(eng_df, use_container_width=True)


# =====================
# TAB 5: Trades
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
            tdf["trade_date"] = tdf["ts"].dt.strftime("%Y-%m-%d")
            tdf["hour"] = tdf["ts"].dt.hour
            tdf["dow"] = tdf["ts"].dt.dayofweek

        total_vol = tdf["amount_usd"].sum() if "amount_usd" in tdf.columns else 0
        total_trades = len(tdf)
        unique_traders = tdf["wallet_address"].nunique() if "wallet_address" in tdf.columns else 0
        avg_trade = total_vol / total_trades if total_trades > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Volume", f"${total_vol:,.0f}")
        col2.metric("Total Trades", f"{total_trades:,}")
        col3.metric("Unique Traders", f"{unique_traders:,}")
        col4.metric("Avg Trade Size", f"${avg_trade:,.0f}")

        # Volume by day
        if "trade_date" in tdf.columns and "amount_usd" in tdf.columns:
            st.subheader("Daily Volume")
            daily_vol = tdf.groupby("trade_date")["amount_usd"].sum()
            st.bar_chart(daily_vol)

        # Volume by token
        if "to_token" in tdf.columns and "amount_usd" in tdf.columns:
            st.subheader("Volume by Token (top 20)")
            vol_by_token = tdf.groupby("to_token")["amount_usd"].sum().sort_values(ascending=False).head(20)
            st.bar_chart(vol_by_token)

        # Trade count by token
        if "to_token" in tdf.columns:
            st.subheader("Trade Count by Token (top 20)")
            count_by_token = tdf["to_token"].value_counts().head(20)
            st.bar_chart(count_by_token)

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

        # Trade source breakdown
        if "source" in tdf.columns:
            st.subheader("Trade Source")
            source_counts = tdf["source"].value_counts()
            st.bar_chart(source_counts)

        # Recent trades table
        st.subheader("Recent Trades")
        display_cols = [c for c in ["timestamp", "wallet_address", "from_token", "to_token", "amount_usd", "status", "source", "type"] if c in tdf.columns]
        recent = tdf[display_cols].sort_values("timestamp", ascending=False).head(50)
        if "wallet_address" in recent.columns:
            recent["wallet_address"] = recent["wallet_address"].apply(short_wallet)
        st.dataframe(recent, use_container_width=True)


# =====================
# TAB 6: Notifications
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
        unique_tappers = tapped_events["wallet_address"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Received", f"{received_count:,}")
        col2.metric("Tapped", f"{tapped_count:,}")
        col3.metric("Tap Rate", f"{tap_rate:.1f}%")
        col4.metric("Unique Tappers", f"{unique_tappers:,}")

        # Tap rate by day
        st.subheader("Tap Rate by Day")
        if not received_events.empty:
            daily_received = received_events.groupby("date").size()
            daily_tapped = tapped_events.groupby("date").size() if not tapped_events.empty else pd.Series(dtype=int)
            daily_rate = (daily_tapped / daily_received * 100).fillna(0)
            st.line_chart(daily_rate)

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
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Time to Trade", f"{avg_seconds:.0f}s")
                    col2.metric("Median Time to Trade", f"{median_seconds:.0f}s")
                    col3.metric("Conversions (tap→trade <1hr)", f"{len(times_to_trade)}")
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

        # Top notification tappers
        if not tapped_events.empty:
            st.subheader("Most Engaged Notification Users")
            top_tappers = tapped_events.groupby("wallet_address").size().sort_values(ascending=False).head(15).reset_index()
            top_tappers.columns = ["wallet_address", "taps"]
            top_tappers["wallet"] = top_tappers["wallet_address"].apply(short_wallet)
            st.dataframe(
                top_tappers[["wallet", "taps"]].rename(columns={"wallet": "Wallet", "taps": "Notification Taps"}),
                use_container_width=True,
            )
