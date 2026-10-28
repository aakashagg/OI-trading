import requests
import time
import json
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Mode Configuration ---
IS_TESTING = False        # True = Replay & Plot NSE Chart API data | False = Live OI Polling
RECORD_LIVE_DATA = True  # Saves live 2.5-min OI snapshots to a file for future backtesting

# --- Strategy Configuration ---
# 10 High-Liquidity Stocks & Indices for OI Tracking
SYMBOLS = ["LTM", "OFSS",  "NIFTY",  "RELIANCE", "INFY", "TCS",  "ITC",  "IDFCFIRSTB", "TATAMOTORS", "BHARTIARTL"]
INDEX_SYMBOLS = {"NIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

POLL_INTERVAL = 120         # 2.5 minutes in seconds
DELAY_BETWEEN_REQUESTS = 2  # 2-second stagger to prevent NSE firewall blocks

# --- Dynamic Threshold Configuration (Last 5-10 Data Points) ---
LOOKBACK_WINDOW = 12      
MIN_SAMPLES = 5          
SPIKE_MULTIPLIER = 5   
MIN_PCT_FLOOR = 5.0      

# --- Backtest Price Filters (Prevents Penny-Option Noise) ---
MIN_OPTION_PRICE = 3.0      # Ignore dead OTM options trading below ₹3.00
MIN_RUPEE_CHANGE = 0.50     # Require at least a ₹0.50 move in 2.5 mins

# --- NSE v3 Endpoints ---
BASE_URL = "https://www.nseindia.com"
OC_PAGE_URL = "https://www.nseindia.com/option-chain"
OPTION_CHAIN_V3_URL = "https://www.nseindia.com/api/option-chain-v3"
CHART_API_URL = "https://www.nseindia.com/api/chart-databyindex"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}

def get_session():
    """Warms up the session on both the homepage and /option-chain to seed NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(BASE_URL, timeout=10)
        session.get(OC_PAGE_URL, timeout=10)
    except Exception as e:
        print(f"Error initializing session: {e}")
    return session

def get_nearest_expiry(session, symbol):
    """Step 1 of NSE v3 API: Fetch the list of expiry dates for the symbol."""
    url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            expiries = data.get("expiryDates") or data.get("records", {}).get("expiryDates", [])
            if expiries:
                return expiries[0]
    except Exception as e:
        print(f"Error fetching expiry for {symbol}: {e}")
    return None

def fetch_option_chain(session, symbol, expiry_date):
    """Step 2 of NSE v3 API: Fetch the option chain for a specific expiry date."""
    instrument_type = "Indices" if symbol.upper() in INDEX_SYMBOLS else "Equities"
    params = {
        "type": instrument_type,
        "symbol": symbol.upper(),
        "expiry": expiry_date
    }
    try:
        response = session.get(OPTION_CHAIN_V3_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code in (401, 403):
            print(f"Session expired or blocked for {symbol}. Refreshing cookies...")
            return None
    except Exception as e:
        print(f"Failed to fetch option chain for {symbol}: {e}")
    return None

def fetch_nse_chart_data(session, identifier):
    """Fetches intraday tick data from the NSE Chart API."""
    try:
        response = session.get(CHART_API_URL, params={"index": identifier}, timeout=10)
        if response.status_code == 200:
            return response.json().get("grapthData", [])
    except Exception:
        pass
    return []

def get_dynamic_5_strikes(records, underlying_value):
    """Dynamically finds the ATM strike and returns the 5 nearest strikes."""
    all_strikes = sorted({r.get("strikePrice") for r in records if r.get("strikePrice") is not None})
    if not all_strikes:
        return None, []
    atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - underlying_value))
    atm_strike = all_strikes[atm_idx]
    start_idx = max(0, atm_idx - 2)
    end_idx = min(len(all_strikes), start_idx + 5)
    return atm_strike, all_strikes[start_idx:end_idx]

def check_dynamic_trigger(pct_chg, history_deque):
    """Compares current % change against the rolling average."""
    if len(history_deque) < MIN_SAMPLES:
        history_deque.append(abs(pct_chg))
        return False, None

    rolling_avg_mag = sum(history_deque) / len(history_deque)
    dynamic_threshold = max(MIN_PCT_FLOOR, rolling_avg_mag * SPIKE_MULTIPLIER)

    is_triggered = abs(pct_chg) >= dynamic_threshold
    history_deque.append(abs(pct_chg))
    return is_triggered, dynamic_threshold

def analyze_oi_changes(symbol, current_data, state, pct_history):
    """Calculates dynamic liquidity filters, overhead walls, trend sync, and alerts."""
    records_block = current_data.get('records', current_data)
    records = records_block.get('data', [])
    underlying_value = records_block.get('underlyingValue', 0)
    timestamp_str = records_block.get('timestamp', datetime.now().strftime('%H:%M:%S'))
    
    if not underlying_value or not records:
        return state

    atm_strike, target_strikes = get_dynamic_5_strikes(records, underlying_value)
    if not atm_strike:
        return state
        
    current_state = {}
    total_ce_oi = 0
    total_pe_oi = 0
    
    # 1. Build Current State, Calculate PCR & Dynamic Base OI
    for record in records:
        strike = record.get('strikePrice')
        if strike in target_strikes:
            ce_oi = record.get('CE', {}).get('openInterest', 0)
            pe_oi = record.get('PE', {}).get('openInterest', 0)
            current_state[strike] = {'CE_OI': ce_oi, 'PE_OI': pe_oi}
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi
            
            if strike not in pct_history:
                pct_history[strike] = {'CE': deque(maxlen=LOOKBACK_WINDOW), 'PE': deque(maxlen=LOOKBACK_WINDOW)}

    window_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
    
    # Dynamic Minimums: Scales perfectly whether the stock averages 1,000 or 150,000 OI
    total_window_oi = total_ce_oi + total_pe_oi
    avg_strike_oi = total_window_oi / 10  # 5 strikes * 2 (CE + PE)
    
    dynamic_min_base = max(50, avg_strike_oi * 0.20)  # Strike must hold 20% of the local average
    dynamic_min_abs = max(10, avg_strike_oi * 0.05)   # Shift must be at least 5% of the local average

    # 2. Overhead Wall Check (Next strike above ATM)
    overhead_wall = False
    if target_strikes:
        try:
            atm_idx = target_strikes.index(atm_strike)
            if atm_idx + 1 < len(target_strikes):
                next_strike = target_strikes[atm_idx + 1]
                wall_ce = current_state.get(next_strike, {}).get('CE_OI', 0)
                wall_pe = current_state.get(next_strike, {}).get('PE_OI', 0)
                if wall_pe > 0 and (wall_ce / wall_pe) >= 2.5:
                    overhead_wall = True
        except ValueError:
            pass

    # 3. Spot Trend Logic
    prev_spot = state.get('prev_spot', underlying_value)
    spot_rising = underlying_value >= prev_spot
    spot_falling = underlying_value <= prev_spot
    prev_strikes = state.get('strikes', {})

    # 4. Evaluate Institutional Filters
    for strike in target_strikes:
        if strike in prev_strikes and strike in current_state:
            prev_ce = prev_strikes[strike]['CE_OI']
            prev_pe = prev_strikes[strike]['PE_OI']
            ce_oi = current_state[strike]['CE_OI']
            pe_oi = current_state[strike]['PE_OI']
            
            ce_pct = ((ce_oi - prev_ce) / prev_ce * 100) if prev_ce > 0 else 0
            pe_pct = ((pe_oi - prev_pe) / prev_pe * 100) if prev_pe > 0 else 0
            ce_abs = ce_oi - prev_ce
            pe_abs = pe_oi - prev_pe
            
            ce_trig, _ = check_dynamic_trigger(ce_pct, pct_history[strike]['CE'])
            pe_trig, _ = check_dynamic_trigger(pe_pct, pct_history[strike]['PE'])
            
            # --- BEARISH CONDITIONS (Spot falling, PCR low) ---
            if spot_falling and window_pcr <= 1.15:
                # CE Spike (Resistance Building)
                if ce_trig and ce_pct > 0 and prev_ce >= dynamic_min_base and abs(ce_abs) >= dynamic_min_abs:
                    print(f"[{timestamp_str}] 🚨 {symbol} RESISTANCE: Strike {strike} CE spiked {ce_pct:+.2f}% (+{ce_abs} lots)")
                # PE Unwinding (Breakdown) - Must be ATM/ITM
                if pe_trig and pe_pct < 0 and strike >= atm_strike and prev_pe >= dynamic_min_base and abs(pe_abs) >= dynamic_min_abs:
                    print(f"[{timestamp_str}] ⚠️ {symbol} BREAKDOWN: Strike {strike} PE dropped {pe_pct:.2f}% ({pe_abs} lots)")

            # --- BULLISH CONDITIONS (Spot rising, PCR high, No Wall) ---
            if spot_rising and window_pcr >= 0.85 and not overhead_wall:
                # PE Spike (Support Building)
                if pe_trig and pe_pct > 0 and prev_pe >= dynamic_min_base and abs(pe_abs) >= dynamic_min_abs:
                    print(f"[{timestamp_str}] 🚨 {symbol} SUPPORT: Strike {strike} PE spiked {pe_pct:+.2f}% (+{pe_abs} lots)")
                # CE Unwinding (Breakout) - Must be ATM/ITM
                if ce_trig and ce_pct < 0 and strike <= atm_strike and prev_ce >= dynamic_min_base and abs(ce_abs) >= dynamic_min_abs:
                    print(f"[{timestamp_str}] ⚠️ {symbol} BREAKOUT: Strike {strike} CE dropped {ce_pct:.2f}% ({ce_abs} lots)")

    if RECORD_LIVE_DATA and not IS_TESTING:
        with open("oi_history.jsonl", "a") as f:
            f.write(json.dumps({
                "timestamp": timestamp_str, 
                "symbol": symbol, 
                "underlyingValue": underlying_value, 
                "strikes": current_state
            }) + "\n")
            
    return {"strikes": current_state, "prev_spot": underlying_value}

def run_nse_chart_backtest(session):
    """Replays data for multiple symbols and saves PNGs instead of opening 10 popup windows."""
    for symbol in SYMBOLS:
        print(f"\n🛠️ Fetching testing data for {symbol}...")
        nearest_expiry = get_nearest_expiry(session, symbol)
        if not nearest_expiry: continue

        data = fetch_option_chain(session, symbol, nearest_expiry)
        if not data: continue

        records_block = data.get('records', data)
        underlying_value = records_block.get('underlyingValue', 0)
        records = records_block.get('data', [])

        atm_strike, target_strikes = get_dynamic_5_strikes(records, underlying_value)
        print(f"[{symbol}] Spot: {underlying_value} | ATM: {atm_strike}")

        chart_series = {}
        for record in records:
            strike = record.get('strikePrice')
            if strike in target_strikes:
                ce_id = record.get('CE', {}).get('identifier')
                pe_id = record.get('PE', {}).get('identifier')
                if ce_id and pe_id:
                    chart_series[strike] = {
                        'CE': fetch_nse_chart_data(session, ce_id),
                        'PE': fetch_nse_chart_data(session, pe_id)
                    }

        sorted_strikes = sorted(chart_series.keys())
        if not sorted_strikes: continue

        fig, axes = plt.subplots(nrows=len(sorted_strikes), ncols=2, figsize=(16, 3.2 * len(sorted_strikes)), sharex=False)
        if len(sorted_strikes) == 1: axes = [axes]
        fig.suptitle(f"{symbol} ({nearest_expiry}) — Dynamic Rolling Triggers", fontsize=13, fontweight="bold")

        for row_idx, strike in enumerate(sorted_strikes):
            for col_idx, opt_type in enumerate(['CE', 'PE']):
                ax = axes[row_idx][col_idx]
                tick_list = chart_series[strike][opt_type]

                if not tick_list: continue

                # Shift NSE timestamps by -19800s (-5.5 hours) so X-axis shows true IST (09:15 to 15:30)
                all_times = [datetime.fromtimestamp((ts_ms / 1000) - 19800) for ts_ms, _ in tick_list]
                all_prices = [price for _, price in tick_list]
                ax.plot(all_times, all_prices, color="#1f77b4", linewidth=1.2, alpha=0.85)

                last_sampled_ts = 0
                prev_price = None
                rolling_history = deque(maxlen=LOOKBACK_WINDOW)
                spike_up_times, spike_up_prices, spike_dn_times, spike_dn_prices = [], [], [], []

                for ts_ms, price in tick_list:
                    if ts_ms - last_sampled_ts >= (POLL_INTERVAL * 1000):
                        # Use the same -19800 offset so trigger markers align directly on the blue line
                        dt_obj = datetime.fromtimestamp((ts_ms / 1000) - 19800)
                        
                        if prev_price and prev_price > 0:
                            pct_chg = ((price - prev_price) / prev_price) * 100
                            is_trig, _ = check_dynamic_trigger(pct_chg, rolling_history)

                            # Only plot trigger if option price is >= ₹3.00 and moved at least ₹0.50
                            if is_trig and prev_price >= MIN_OPTION_PRICE and abs(price - prev_price) >= MIN_RUPEE_CHANGE:
                                if pct_chg > 0:
                                    spike_up_times.append(dt_obj); spike_up_prices.append(price)
                                else:
                                    spike_dn_times.append(dt_obj); spike_dn_prices.append(price)
                                    
                        prev_price = price
                        last_sampled_ts = ts_ms

                if spike_up_times: ax.scatter(spike_up_times, spike_up_prices, color="limegreen", marker="^", s=70, zorder=5)
                if spike_dn_times: ax.scatter(spike_dn_times, spike_dn_prices, color="crimson", marker="v", s=70, zorder=5)
                ax.set_title(f"{strike} {opt_type} | Triggers: {len(spike_up_times) + len(spike_dn_times)}", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_file = f"{symbol.lower()}_dynamic_triggers.png"
        plt.savefig(output_file, dpi=150)
        plt.close(fig) 
        print(f"📊 Chart saved: {output_file}")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)

def main():
    session = get_session()
    
    if IS_TESTING:
        print(f"▶️ Generating backtest charts for {len(SYMBOLS)} symbols...")
        run_nse_chart_backtest(session)
        print("\n✅ All charts generated successfully. Check your folder for the PNG files.")
    else:
        print(f"🟢 Starting Live NSE Option Chain OI Monitor for {len(SYMBOLS)} symbols...")
        
        states = {sym: {"strikes": {}, "prev_spot": 0.0} for sym in SYMBOLS}
        pct_histories = {sym: {} for sym in SYMBOLS}
        nearest_expiries = {sym: None for sym in SYMBOLS}
        
        while True:
            for symbol in SYMBOLS:
                if not nearest_expiries[symbol]:
                    nearest_expiries[symbol] = get_nearest_expiry(session, symbol)
                    
                if nearest_expiries[symbol]:
                    data = fetch_option_chain(session, symbol, nearest_expiries[symbol])
                    if data:
                        states[symbol] = analyze_oi_changes(symbol, data, states[symbol], pct_histories[symbol])
                    else:
                        session = get_session() 
                
                time.sleep(DELAY_BETWEEN_REQUESTS) 
                
            print(f"\nCycle complete. Sleeping for {POLL_INTERVAL} seconds...\n")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
