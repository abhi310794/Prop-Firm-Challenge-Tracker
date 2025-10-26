import streamlit as st
import json
import os
import math
from datetime import date, timedelta, datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION - Settings that control the app
# ============================================================================
st.set_page_config(
    page_title="Prop Firm Trading Journal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS - Values that never change (easier to modify later)
# ============================================================================
DATA_DIR = "data_files"
DEFAULT_ACCOUNT_SIZE = 200000.0
DEFAULT_BALANCE = 200000.0
DEFAULT_TARGET_PCT = 4.0
DEFAULT_DAYS_TO_PASS = 63
DEFAULT_DAILY_DRAWDOWN_PCT = 10.0
DEFAULT_RISK_PER_TRADE_PCT = 0.25

PROP_FIRMS = [
    "Atlas Funded",
    "Audacity Capital", 
    "E8 Markets",
    "FTM Traders",
    "Instant Funding",
    "Other"
]

# Mood emoji options
MOOD_OPTIONS = {
    "😊 Confident": "😊",
    "😐 Neutral": "😐",
    "😰 Nervous": "😰",
    "😤 Frustrated": "😤",
    "🤩 Excellent": "🤩",
    "😴 Tired": "😴",
    "🤔 Uncertain": "🤔",
    "💪 Strong": "💪"
}

# Color constants for consistent styling
COLOR_PROFIT = "#16A34A"  # Green
COLOR_LOSS = "#EF4444"    # Red
COLOR_WARNING = "#FFD700"  # Gold
COLOR_NEUTRAL = "#FFFFFF"  # White
COLOR_OFF_WHITE = "#D1D5DB"  # Off-white for secondary text
COLOR_HIGHLIGHT = "#22C55E" # Bright green

# ============================================================================
# UTILITY FUNCTIONS - Helper functions that do ONE thing well
# ============================================================================

def sanitize_filename(firm_name: str) -> str:
    """
    Convert firm name to safe filename.
    Example: "Atlas Funded" -> "atlas_funded.json"
    
    Why: File systems don't like spaces and special characters
    """
    safe_name = firm_name.replace(" ", "_").lower()
    # Remove any potentially dangerous characters
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    return os.path.join(DATA_DIR, f"{safe_name}.json")


def get_daily_entries_filename(firm_name: str) -> str:
    """Get filename for daily trading entries"""
    safe_name = firm_name.replace(" ", "_").lower()
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    return os.path.join(DATA_DIR, f"{safe_name}_daily_entries.json")


def load_firm_data(firm_name: str) -> Dict[str, Any]:
    """
    Load data for a specific firm from JSON file.
    Returns empty dict if file doesn't exist or is corrupted.
    
    Why: We cache this so we don't reload the file on every widget interaction
    """
    filepath = sanitize_filename(firm_name)
    
    if not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except (json.JSONDecodeError, IOError) as e:
        # If file is corrupted, log error and return empty dict
        st.error(f"⚠️ Could not load data for {firm_name}: {str(e)}")
        return {}


def load_daily_entries(firm_name: str) -> List[Dict[str, Any]]:
    """Load daily trading entries for a firm"""
    filepath = get_daily_entries_filename(firm_name)
    
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)
            # Sort by date, newest first
            entries.sort(key=lambda x: x.get("date", ""), reverse=True)
            return entries
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"⚠️ Could not load daily entries: {str(e)}")
        return []


def save_firm_data(firm_name: str, data: Dict[str, Any]) -> bool:
    """
    Save firm data to JSON file.
    Returns True if successful, False if failed.
    
    Why: Return success/failure so we can handle errors properly
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filepath = sanitize_filename(firm_name)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            # indent=2 makes the JSON human-readable
            json.dump(data, f, indent=2, default=str)
        return True
    except IOError as e:
        st.error(f"❌ Failed to save data: {str(e)}")
        return False


def save_daily_entries(firm_name: str, entries: List[Dict[str, Any]]) -> bool:
    """Save daily trading entries"""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = get_daily_entries_filename(firm_name)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        return True
    except IOError as e:
        st.error(f"❌ Failed to save daily entries: {str(e)}")
        return False


def add_daily_entry(firm_name: str, entry_date: date, pnl: float, mood: str) -> bool:
    """Add or update a daily entry"""
    entries = load_daily_entries(firm_name)
    
    # Check if entry for this date already exists
    date_str = entry_date.isoformat()
    existing_index = None
    
    for i, entry in enumerate(entries):
        if entry.get("date") == date_str:
            existing_index = i
            break
    
    new_entry = {
        "date": date_str,
        "pnl": pnl,
        "mood": mood,
        "timestamp": datetime.now().isoformat()
    }
    
    if existing_index is not None:
        # Update existing entry
        entries[existing_index] = new_entry
    else:
        # Add new entry
        entries.append(new_entry)
    
    return save_daily_entries(firm_name, entries)


def delete_daily_entry(firm_name: str, entry_date: str) -> bool:
    """Delete a daily entry by date"""
    entries = load_daily_entries(firm_name)
    
    # Filter out the entry with matching date
    entries = [e for e in entries if e.get("date") != entry_date]
    
    return save_daily_entries(firm_name, entries)


def validate_positive_number(value: float, field_name: str) -> Optional[str]:
    """
    Check if a number is positive.
    Returns error message if invalid, None if valid.
    
    Why: Keep validation logic in one place
    """
    if value <= 0:
        return f"❌ {field_name} must be greater than 0"
    return None


def validate_percentage(value: float, field_name: str) -> Optional[str]:
    """
    Check if a percentage is reasonable (0-100).
    Returns error message if invalid, None if valid.
    """
    if value < 0 or value > 100:
        return f"❌ {field_name} must be between 0 and 100"
    return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide two numbers safely without crashing.
    Returns default value if denominator is zero.
    
    Why: Prevent "division by zero" errors that crash the app
    """
    if denominator == 0:
        return default
    return numerator / denominator


# ============================================================================
# CALCULATION FUNCTIONS - Pure functions that just do math
# ============================================================================

def calculate_target_dollars(account_size: float, target_pct: float) -> float:
    """How much money needed to hit target percentage"""
    return account_size * (target_pct / 100.0)


def calculate_overall_profit(balance: float, account_size: float) -> float:
    """Current profit or loss (negative = loss)"""
    return balance - account_size


def calculate_profit_percentage(profit: float, account_size: float) -> float:
    """Profit as percentage of account size"""
    return safe_divide(profit, account_size) * 100.0


def calculate_remaining_to_target(target_dollars: float, current_profit: float) -> float:
    """How much more profit needed to hit target"""
    return max(0, target_dollars - current_profit)


def calculate_daily_profit_needed(remaining: float, days_left: int) -> float:
    """How much profit needed per day to hit target"""
    if days_left <= 0:
        return float('inf')  # Impossible - time's up!
    return remaining / days_left


def calculate_risk_per_trade(balance: float, risk_pct: float) -> float:
    """Dollar amount risked per trade"""
    return balance * (risk_pct / 100.0)


def calculate_trades_until_breach(balance: float, breach_level: float, risk_per_trade: float) -> float:
    """How many losing trades before hitting max loss"""
    if risk_per_trade <= 0:
        return float('inf')
    
    buffer = balance - breach_level
    if buffer <= 0:
        return 0  # Already at or past breach!
    
    return math.floor(buffer / risk_per_trade)


def calculate_days_remaining(end_date: date, today: date) -> int:
    """Days left in challenge"""
    return max(0, (end_date - today).days)


def calculate_progress(current: float, target: float) -> float:
    """Progress as decimal between 0.0 and 1.0"""
    if target <= 0:
        return 0.0
    progress = current / target
    return max(0.0, min(1.0, progress))  # Clamp between 0 and 1


def create_cumulative_pnl_chart(entries: List[Dict[str, Any]], days: int, title: str):
    """
    Create a cumulative P/L line chart for specified number of days
    
    Why: Visual representation of account growth over time
    """
    if not entries:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#9CA3AF")
        )
        fig.update_layout(
            title=title,
            height=400,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1E293B",
            font=dict(color="#FFFFFF")
        )
        return fig
    
    # Filter entries for the specified time period
    cutoff_date = date.today() - timedelta(days=days-1)
    filtered_entries = []
    
    for entry in entries:
        entry_date_obj = datetime.fromisoformat(entry["date"]).date()
        if entry_date_obj >= cutoff_date:
            filtered_entries.append(entry)
    
    if not filtered_entries:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for last {days} days",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#9CA3AF")
        )
        fig.update_layout(
            title=title,
            height=400,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1E293B",
            font=dict(color="#FFFFFF")
        )
        return fig
    
    # Sort by date (oldest first for cumulative calculation)
    filtered_entries.sort(key=lambda x: x["date"])
    
    # Calculate cumulative P/L
    dates = []
    cumulative_pnl = []
    running_total = 0
    
    for entry in filtered_entries:
        dates.append(datetime.fromisoformat(entry["date"]).date())
        running_total += entry["pnl"]
        cumulative_pnl.append(running_total)
    
    # Determine line color based on final P/L
    line_color = COLOR_PROFIT if running_total >= 0 else COLOR_LOSS
    
    # Create the chart
    fig = go.Figure()
    
    # Add the cumulative P/L line
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P/L',
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color),
        fill='tozeroy',
        fillcolor=f'rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.2)'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#6B7280", line_width=1)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#FFFFFF")),
        xaxis_title="Date",
        yaxis_title="Cumulative P/L ($)",
        height=400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1E293B",
        font=dict(color="#FFFFFF", size=14),
        hovermode='x unified',
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor="#374151",
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#374151",
            gridwidth=1,
            tickprefix="$",
            tickformat=",.0f"
        )
    )
    
    return fig


# ============================================================================
# UI COMPONENT FUNCTIONS - Reusable UI pieces
# ============================================================================

def render_metric_card(title: str, value: str, color: str = COLOR_NEUTRAL):
    """
    Render a styled metric card with larger text.
    
    Why: Don't repeat the same HTML 5 times
    """
    st.markdown(f"""
        <div style='
            border: 2px solid {COLOR_NEUTRAL};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            background-color: #1E293B;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        '>
            <h4 style='margin: 0 0 12px 0; color: {COLOR_NEUTRAL}; font-size: 20px; font-weight: 600;'>{title}</h4>
            <p style='
                font-size: 24px;
                font-weight: 700;
                margin: 0;
                color: {color};
                line-height: 1.4;
            '>{value}</p>
        </div>
    """, unsafe_allow_html=True)


def format_currency(amount: float, show_sign: bool = False) -> str:
    """
    Format number as currency with commas.
    Example: 1234.56 -> "$1,234.56" or "$1,235" (rounded)
    """
    if show_sign:
        return f"${amount:+,.0f}"
    return f"${amount:,.0f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage. Example: 0.05 -> "5.00%"""
    return f"{value * 100:.{decimals}f}%"


# ============================================================================
# MAIN APP INITIALIZATION
# ============================================================================

# Initialize session state for selected firm if not exists
if "selected_firm" not in st.session_state:
    st.session_state.selected_firm = PROP_FIRMS[0]

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# HEADER WITH FIRM SELECTOR
# ============================================================================

# Custom CSS for better styling
st.markdown("""
<style>
    /* Make dropdown larger and more visible */
    [data-testid='stSelectbox'] div[role='combobox'] {
        color: #FFD700;
        font-weight: 700;
        font-size: 20px;
        min-height: 45px;
    }
    
    /* Reduce padding around elements */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Style progress bars */
    .stProgress > div > div > div {
        background-color: #22C55E;
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header with firm selector - using container for vertical alignment
header_container = st.container()

with header_container:
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.markdown(f"""
            <h1 style='color: white; font-size: 42px; margin: 0; line-height: 1.2;'>
                📊 {st.session_state.selected_firm}
            </h1>
        """, unsafe_allow_html=True)

    with header_col2:
        # Add some top margin to align with the header text
        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
        selected_firm = st.selectbox(
            "Select Firm",
            PROP_FIRMS,
            index=PROP_FIRMS.index(st.session_state.selected_firm),
            key="firm_selector",
            label_visibility="collapsed"
        )
    
    # Update session state if firm changed
    if selected_firm != st.session_state.selected_firm:
        st.session_state.selected_firm = selected_firm
        st.rerun()  # Refresh to load new firm data

# ============================================================================
# TABS - Main navigation
# ============================================================================

tab1, tab2 = st.tabs(["📊 Dashboard", "📝 Daily Tracker"])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================

with tab1:
    # ========================================================================
    # LOAD DATA (with caching to prevent reloading on every interaction)
    # ========================================================================

    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_firm_data(firm_name: str) -> Dict[str, Any]:
        """Cached data loader - only loads from disk when needed"""
        return load_firm_data(firm_name)

    firm_data = get_firm_data(st.session_state.selected_firm)

    # ========================================================================
    # SIDEBAR - INPUT FORM
    # ========================================================================

    st.sidebar.title("⚙️ Account Settings")

    st.sidebar.subheader("📊 Account Details")

    # Account inputs with validation
    account_size = st.sidebar.number_input(
        "Account Size ($)",
        min_value=0.0,
        value=float(firm_data.get("account_size", DEFAULT_ACCOUNT_SIZE)),
        step=1000.0,
        help="Total account size provided by prop firm"
    )

    balance = st.sidebar.number_input(
        "Current Balance ($)",
        min_value=0.0,
        value=float(firm_data.get("balance", DEFAULT_BALANCE)),
        step=100.0,
        help="Your current account balance"
    )

    target_pct = st.sidebar.number_input(
        "Target Profit %",
        min_value=0.0,
        max_value=100.0,
        value=float(firm_data.get("target_pct", DEFAULT_TARGET_PCT)),
        step=0.5,
        help="Profit percentage needed to pass challenge"
    )

    # Date inputs
    start_date_str = firm_data.get("challenge_start_date")
    if start_date_str:
        try:
            default_start = datetime.fromisoformat(start_date_str).date()
        except (ValueError, TypeError):
            default_start = date.today()
    else:
        default_start = date.today()

    challenge_start_date = st.sidebar.date_input(
        "Challenge Start Date",
        value=default_start,
        help="When your challenge period started"
    )

    days_to_pass = st.sidebar.number_input(
        "Challenge Duration (Days)",
        min_value=1,
        max_value=365,
        value=int(firm_data.get("days_to_pass", DEFAULT_DAYS_TO_PASS)),
        step=1,
        help="Total days allowed to complete challenge"
    )

    st.sidebar.subheader("⚔️ Risk Management")

    daily_drawdown_pct = st.sidebar.number_input(
        "Daily Drawdown Limit %",
        min_value=0.0,
        max_value=100.0,
        value=float(firm_data.get("daily_drawdown_pct", DEFAULT_DAILY_DRAWDOWN_PCT)),
        step=0.5,
        help="Maximum daily loss allowed"
    )

    max_loss_breach_level = st.sidebar.number_input(
        "Max Loss Breach Level ($)",
        min_value=0.0,
        value=float(firm_data.get("max_loss_equity_breach_level", account_size * 0.9)),
        step=100.0,
        help="Account balance that triggers rule violation"
    )

    risk_per_trade_pct = st.sidebar.number_input(
        "Risk Per Trade %",
        min_value=0.0,
        max_value=10.0,
        value=float(firm_data.get("risk_per_trade_pct", DEFAULT_RISK_PER_TRADE_PCT)),
        step=0.05,
        help="Percentage of balance risked per trade"
    )

    # ========================================================================
    # VALIDATION - Check for errors before calculating
    # ========================================================================

    errors = []

    # Validate all inputs
    error = validate_positive_number(account_size, "Account Size")
    if error:
        errors.append(error)

    error = validate_positive_number(balance, "Balance")
    if error:
        errors.append(error)

    error = validate_percentage(target_pct, "Target Profit %")
    if error:
        errors.append(error)

    error = validate_percentage(daily_drawdown_pct, "Daily Drawdown %")
    if error:
        errors.append(error)

    error = validate_percentage(risk_per_trade_pct, "Risk Per Trade %")
    if error:
        errors.append(error)

    # Check if breach level makes sense
    if max_loss_breach_level >= balance:
        errors.append("⚠️ Max loss breach level should be below current balance")

    # Display errors if any
    if errors:
        st.sidebar.error("Please fix the following errors:")
        for err in errors:
            st.sidebar.write(err)

    # ========================================================================
    # SAVE BUTTON
    # ========================================================================

    st.sidebar.markdown("---")

    if st.sidebar.button("💾 Save Settings", type="primary", use_container_width=True):
        if errors:
            st.sidebar.error("❌ Cannot save - please fix errors above")
        else:
            # Prepare data to save
            save_data = {
                "account_size": account_size,
                "balance": balance,
                "target_pct": target_pct,
                "challenge_start_date": challenge_start_date.isoformat(),
                "days_to_pass": days_to_pass,
                "daily_drawdown_pct": daily_drawdown_pct,
                "max_loss_equity_breach_level": max_loss_breach_level,
                "risk_per_trade_pct": risk_per_trade_pct,
                "last_saved": datetime.now().isoformat()
            }
            
            # Save and show result
            if save_firm_data(st.session_state.selected_firm, save_data):
                st.sidebar.success(f"✅ Settings saved for {st.session_state.selected_firm}!")
                # Clear cache to reload fresh data
                get_firm_data.clear()
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to save settings")

    # Show last saved date if available
    last_saved = firm_data.get("last_saved")
    if last_saved:
        try:
            saved_date = datetime.fromisoformat(last_saved)
            st.sidebar.caption(f"Last saved: {saved_date.strftime('%Y-%m-%d %H:%M')}")
        except (ValueError, TypeError):
            pass

    # ========================================================================
    # CALCULATIONS - Only run if no validation errors
    # ========================================================================

    if not errors:
        # Calculate all metrics
        target_dollars = calculate_target_dollars(account_size, target_pct)
        overall_profit = calculate_overall_profit(balance, account_size)
        profit_pct = calculate_profit_percentage(overall_profit, account_size)
        remaining_to_target = calculate_remaining_to_target(target_dollars, overall_profit)
        
        challenge_end_date = challenge_start_date + timedelta(days=days_to_pass)
        days_remaining = calculate_days_remaining(challenge_end_date, date.today())
        
        daily_profit_needed = calculate_daily_profit_needed(remaining_to_target, days_remaining)
        risk_per_trade = calculate_risk_per_trade(balance, risk_per_trade_pct)
        trades_until_breach = calculate_trades_until_breach(balance, max_loss_breach_level, risk_per_trade)
        
        target_progress = calculate_progress(overall_profit, target_dollars)
        
        # Calculate how much drawdown is being used (as percentage of account)
        drawdown_used_pct = abs(profit_pct) / 100.0
        
        # ====================================================================
        # KEY INSIGHT SECTION - Two big important numbers
        # ====================================================================
        
        st.markdown("---")
        
        # Format trades display
        trades_display = "∞" if trades_until_breach == float('inf') else f"{int(trades_until_breach):,}"
        
        # Format daily profit needed
        if daily_profit_needed == float('inf'):
            daily_display = "TIME'S UP!"
            daily_color = COLOR_LOSS
        else:
            daily_display = f"{format_currency(daily_profit_needed)}"
            daily_color = COLOR_HIGHLIGHT
        
        # Trades color based on danger level
        if trades_until_breach == 0:
            trades_color = COLOR_LOSS
        elif trades_until_breach < 10:
            trades_color = COLOR_LOSS
        elif trades_until_breach < 20:
            trades_color = COLOR_WARNING
        else:
            trades_color = COLOR_PROFIT
        
        # Create two columns for the key metrics
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
                    padding: 30px;
                    border-radius: 12px;
                    border-left: 5px solid {daily_color};
                    text-align: center;
                '>
                    <h2 style='margin: 0 0 10px 0; font-size: 48px; font-weight: 700;'>
                        📈 <span style='color: {daily_color};'>{daily_display}/day</span> <span style='color: white;'>for {days_remaining} days</span>
                    </h2>
                    <p style='color: white; font-size: 24px; margin: 0; font-weight: 500;'>
                        Risk {format_currency(risk_per_trade)} ({risk_per_trade_pct}%) per trade
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            # Calculate buffer remaining
            buffer_remaining = balance - max_loss_breach_level
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
                    padding: 30px;
                    border-radius: 12px;
                    border-left: 5px solid {trades_color};
                    text-align: center;
                '>
                    <h2 style='margin: 0 0 10px 0; font-size: 48px; font-weight: 700;'>
                        <span style='color: {trades_color};'>⚔️ {trades_display}</span> <span style='color: white;'>Trades left</span>
                    </h2>
                    <p style='color: white; font-size: 24px; margin: 0; font-weight: 500;'>
                        {format_currency(buffer_remaining)} buffer remaining
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # METRICS CARDS - Main dashboard
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Account Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Balance color: green if above account size, red if below
            balance_color = COLOR_PROFIT if balance >= account_size else COLOR_LOSS
            balance_display = f"<span style='color:{balance_color};'>{format_currency(balance)}</span> / <span style='color:{COLOR_OFF_WHITE};'>{format_currency(account_size)}</span>"
            render_metric_card(
                "💰 Balance",
                balance_display,
                COLOR_NEUTRAL  # Using neutral for the border/background
            )
        
        with col2:
            profit_color = COLOR_PROFIT if overall_profit >= 0 else COLOR_LOSS
            profit_display = f"<span style='color:{profit_color};'>{format_percentage(profit_pct / 100, 2)}</span> / <span style='color:{profit_color};'>{format_currency(overall_profit, show_sign=True)}</span>"
            render_metric_card(
                "📈 Profit / Loss",
                profit_display,
                COLOR_NEUTRAL
            )
        
        with col3:
            remaining_pct = safe_divide(remaining_to_target, account_size) * 100
            target_color = COLOR_PROFIT if remaining_to_target == 0 else COLOR_NEUTRAL
            target_display = f"{format_currency(remaining_to_target)} / {remaining_pct:.2f}%"
            render_metric_card(
                "🎯 Target Remaining",
                target_display,
                target_color
            )
        
        with col4:
            risk_color = COLOR_WARNING if trades_until_breach < 20 else COLOR_NEUTRAL
            risk_display = f"{format_currency(risk_per_trade)} / {trades_display}"
            render_metric_card(
                "⚔️ Risk Per Trade",
                risk_display,
                risk_color
            )
        
        with col5:
            days_color = COLOR_LOSS if days_remaining < 7 else COLOR_WARNING if days_remaining < 14 else COLOR_NEUTRAL
            time_display = f"{days_remaining} days / {format_currency(daily_profit_needed)}"
            render_metric_card(
                "📆 Time Remaining",
                time_display,
                days_color
            )
        
        # ====================================================================
        # PROGRESS BARS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🏁 Progress Tracking")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Target Progress**")
            st.progress(target_progress)
            progress_text = f"{format_percentage(target_progress, 1)} complete"
            if target_progress >= 1.0:
                st.success(f"✅ {progress_text} - Target achieved!")
            elif target_progress >= 0.75:
                st.info(f"📊 {progress_text} - Almost there!")
            else:
                st.write(f"📊 {progress_text}")
        
        with col_right:
            st.markdown("**Account Drawdown Used**")
            # Show how much of the account has moved (up or down)
            st.progress(min(drawdown_used_pct, 1.0))
            if overall_profit >= 0:
                st.success(f"✅ Account up {format_percentage(profit_pct / 100, 2)}")
            else:
                st.warning(f"⚠️ Account down {format_percentage(abs(profit_pct) / 100, 2)}")

    else:
        # Show placeholder if there are validation errors
        st.warning("⚠️ Please fix the validation errors in the sidebar to see your dashboard")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("---")
    st.caption(f"📅 Tracking {st.session_state.selected_firm} • Updated {date.today().strftime('%B %d, %Y')}")
    st.caption("💾 Data saved locally in `data_files/` folder • Built with Streamlit")
    st.caption("💡 **Tip**: All changes are auto-saved when you click the Save button")

# ============================================================================
# TAB 2: DAILY TRACKER
# ============================================================================

with tab2:
    st.markdown("## 📝 Daily Trading Journal")
    st.markdown(f"Track your daily performance for **{st.session_state.selected_firm}**")
    
    st.markdown("---")
    
    # ========================================================================
    # INPUT FORM
    # ========================================================================
    
    st.markdown("### ➕ Add Daily Entry")
    
    col_input1, col_input2, col_input3, col_input4 = st.columns([2, 2, 2, 1])
    
    with col_input1:
        entry_date = st.date_input(
            "📅 Date",
            value=date.today(),
            max_value=date.today(),
            help="Select the trading day"
        )
    
    with col_input2:
        daily_pnl = st.number_input(
            "💰 P/L ($)",
            value=0.0,
            step=10.0,
            format="%.2f",
            help="Enter profit (positive) or loss (negative)"
        )
    
    with col_input3:
        mood_selection = st.selectbox(
            "😊 Mood",
            options=list(MOOD_OPTIONS.keys()),
            help="How did you feel during trading?"
        )
    
    with col_input4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("💾 Save Entry", type="primary", use_container_width=True):
            mood_emoji = MOOD_OPTIONS[mood_selection]
            if add_daily_entry(st.session_state.selected_firm, entry_date, daily_pnl, mood_emoji):
                st.success(f"✅ Entry saved for {entry_date.strftime('%Y-%m-%d')}")
                st.rerun()
            else:
                st.error("❌ Failed to save entry")
    
    st.markdown("---")
    
    # ========================================================================
    # DISPLAY ENTRIES - Last 30 days
    # ========================================================================
    
    # Load entries
    entries = load_daily_entries(st.session_state.selected_firm)
    
    if not entries:
        st.info("📝 No entries yet. Add your first daily entry above!")
    else:
        # ====================================================================
        # CUMULATIVE P/L CHARTS - 7 days and 30 days side by side
        # ====================================================================
        
        st.markdown("### 📈 Performance Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_7day = create_cumulative_pnl_chart(entries, 7, "Last 7 Days - Cumulative P/L")
            st.plotly_chart(fig_7day, use_container_width=True)
        
        with chart_col2:
            fig_30day = create_cumulative_pnl_chart(entries, 30, "Last 30 Days - Cumulative P/L")
            st.plotly_chart(fig_30day, use_container_width=True)
        
        st.markdown("---")
        
        # ====================================================================
        # SUMMARY STATS
        # ====================================================================
        
        # Filter to last 30 days for stats
        thirty_days_ago = date.today() - timedelta(days=30)
        recent_entries = []
        
        for entry in entries:
            entry_date_obj = datetime.fromisoformat(entry["date"]).date()
            if entry_date_obj >= thirty_days_ago:
                recent_entries.append(entry)
        
        if recent_entries:
            # Calculate stats
            total_pnl = sum(e["pnl"] for e in recent_entries)
            winning_days = sum(1 for e in recent_entries if e["pnl"] > 0)
            losing_days = sum(1 for e in recent_entries if e["pnl"] < 0)
            break_even_days = sum(1 for e in recent_entries if e["pnl"] == 0)
            
            avg_win = sum(e["pnl"] for e in recent_entries if e["pnl"] > 0) / winning_days if winning_days > 0 else 0
            avg_loss = sum(e["pnl"] for e in recent_entries if e["pnl"] < 0) / losing_days if losing_days > 0 else 0
            
            # Stats cards
            st.markdown("### 📊 Summary Stats (Last 30 Days)")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                pnl_color = COLOR_PROFIT if total_pnl >= 0 else COLOR_LOSS
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #1E293B; border-radius: 8px;'>
                        <h4 style='color: #D1D5DB; margin: 0; font-size: 14px;'>Total P/L</h4>
                        <p style='color: {pnl_color}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;'>
                            {format_currency(total_pnl, show_sign=True)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_col2:
                win_rate = (winning_days / len(recent_entries) * 100) if recent_entries else 0
                win_color = COLOR_PROFIT if win_rate >= 50 else COLOR_LOSS
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #1E293B; border-radius: 8px;'>
                        <h4 style='color: #D1D5DB; margin: 0; font-size: 14px;'>Win Rate</h4>
                        <p style='color: {win_color}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;'>
                            {win_rate:.1f}%
                        </p>
                        <p style='color: #9CA3AF; font-size: 12px; margin: 5px 0 0 0;'>
                            {winning_days}W / {losing_days}L
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #1E293B; border-radius: 8px;'>
                        <h4 style='color: #D1D5DB; margin: 0; font-size: 14px;'>Avg Win</h4>
                        <p style='color: {COLOR_PROFIT}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;'>
                            {format_currency(avg_win, show_sign=True)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #1E293B; border-radius: 8px;'>
                        <h4 style='color: #D1D5DB; margin: 0; font-size: 14px;'>Avg Loss</h4>
                        <p style='color: {COLOR_LOSS}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;'>
                            {format_currency(avg_loss, show_sign=True)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display table
            st.markdown("### 📋 Entry History (Last 30 Days)")
            
            # Create DataFrame for better display
            df_data = []
            for entry in recent_entries:
                entry_date_obj = datetime.fromisoformat(entry["date"]).date()
                pnl = entry["pnl"]
                mood = entry.get("mood", "😐")
                
                df_data.append({
                    "Date": entry_date_obj.strftime("%Y-%m-%d"),
                    "Day": entry_date_obj.strftime("%A"),
                    "P/L": format_currency(pnl, show_sign=True),
                    "Mood": mood,
                    "Result": "✅ Win" if pnl > 0 else "❌ Loss" if pnl < 0 else "➖ Break Even"
                })
            
            df = pd.DataFrame(df_data)
            
            # Display with custom styling
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Delete functionality
            st.markdown("---")
            st.markdown("### 🗑️ Manage Entries")
            
            delete_col1, delete_col2 = st.columns([3, 1])
            
            with delete_col1:
                dates_to_delete = [datetime.fromisoformat(e["date"]).date().strftime("%Y-%m-%d") for e in recent_entries]
                selected_date = st.selectbox(
                    "Select entry to delete",
                    options=dates_to_delete,
                    help="Choose a date to delete its entry"
                )
            
            with delete_col2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    if delete_daily_entry(st.session_state.selected_firm, selected_date):
                        st.success(f"✅ Entry for {selected_date} deleted")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete entry")
        else:
            st.info("📝 No entries in the last 30 days")
    
    # Footer
    st.markdown("---")
    st.caption(f"💡 **Tip**: Track your mood to identify patterns between emotions and performance")
    st.caption(f"📁 Entries saved in: `data_files/{st.session_state.selected_firm.lower().replace(' ', '_')}_daily_entries.json`")