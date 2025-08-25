import math
import time
from datetime import datetime

import pandas as pd
import pytz
import requests
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard")

# =========================
# Auto-refresh every 10 minutes (frontend trigger)
# =========================
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0

# 600,000 ms = 10 minutes
_ = st_autorefresh(interval=600_000, key="auto_refresh")

# =========================
# Backend guard for snapshots (~10 minutes)
# =========================
if "last_snapshot_time" not in st.session_state:
    st.session_state["last_snapshot_time"] = time.time()

def auto_snapshot_due():
    now = time.time()
    if now - st.session_state["last_snapshot_time"] >= 600:
        st.session_state["last_snapshot_time"] = now
        return True
    return False

# Manual refresh
refresh_btn = st.sidebar.button("Refresh Now")

# =========================
# Constants
# =========================
IST = pytz.timezone("Asia/Kolkata")
RISK_FREE = 0.06
DIV_YIELD = 0.00

# =========================
# NSE client
# =========================
class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
        })

    def _warm_cookies(self):
        try:
            self.session.get("https://www.nseindia.com/", timeout=10)
        except Exception:
            pass

    def get_json(self, url, max_tries=5, sleep_between=1.4):
        last_err = None
        for _ in range(max_tries):
            try:
                self._warm_cookies()
                resp = self.session.get(url, timeout=12)
                if resp.status_code in (401, 403):
                    last_err = Exception(f"HTTP {resp.status_code}")
                    time.sleep(sleep_between)
                    continue
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(sleep_between)
        st.error(f"NSE API error: {last_err}")
        return None

nse = NSEClient()

# =========================
# Helpers
# =========================
def format_inr(value: float) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if v >= 1e7:
        return f"{v/1e7:.2f} Cr"
    elif v >= 1e5:
        return f"{v/1e5:.2f} Lakh"
    else:
        return f"{v:.0f}"

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_d1_d2(S, K, r, q, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return None, None
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_delta_gamma(S, K, r, q, sigma, T, option_type: str):
    """
    Black–Scholes delta & gamma. option_type in {"C","P"}
    """
    d1, d2 = bs_d1_d2(S, K, r, q, sigma, T)
    if d1 is None:
        return 0.0, 0.0
    # Delta
    if option_type == "C":
        delta = math.exp(-q * T) * norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm_cdf(-d1)
    # Gamma (same for C/P)
    gamma = (math.exp(-q * T) * norm_pdf(d1)) / (S * sigma * math.sqrt(T))
    return delta, gamma

def nearest_index(sorted_list, x):
    if not sorted_list:
        return None
    return min(range(len(sorted_list)), key=lambda i: abs(sorted_list[i] - x))

def parse_expiry_to_dt(expiry_str: str):
    """
    Convert NSE expiry string to timezone-aware datetime at 15:30 IST.
    """
    fmts = ["%d-%b-%Y", "%d-%b-%y"]
    dt_naive = None
    for f in fmts:
        try:
            dt_naive = datetime.strptime(expiry_str, f)
            break
        except Exception:
            continue
    if dt_naive is None:
        return None
    return IST.localize(dt_naive.replace(hour=15, minute=30, second=0, microsecond=0))

def time_to_expiry_years(expiry_str: str) -> float:
    expiry_dt = parse_expiry_to_dt(expiry_str)
    if expiry_dt is None:
        return 1.0 / 365.0
    now_ist = datetime.now(IST)
    sec = (expiry_dt - now_ist).total_seconds()
    if sec <= 0:
        return 1.0 / (365.0 * 24.0 * 3600.0)
    return sec / (365.0 * 24.0 * 3600.0)

def fetch_option_chain(symbol: str):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    return nse.get_json(url)

# =========================
# Core computations
# =========================
def compute_metrics(data: dict, symbol: str, expiry: str):
    recs = (data or {}).get("records", {})
    rows = recs.get("data", []) or []
    spot = recs.get("underlyingValue")
    if spot is None:
        return None

    T = time_to_expiry_years(expiry)
    r = RISK_FREE
    q = DIV_YIELD

    ce_rows, pe_rows, strikes_all = [], [], []
    for item in rows:
        if item.get("expiryDate") != expiry:
            continue
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
        strikes_all.append(strike)
        if ce:
            ce_rows.append({
                "strike": strike,
                "iv": (ce.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": ce.get("openInterest", 0) or 0,
                "bidQty": ce.get("bidQty", 0) or 0,
                "ltp": ce.get("lastPrice", 0.0) or 0.0,
            })
        if pe:
            pe_rows.append({
                "strike": strike,
                "iv": (pe.get("impliedVolatility", 0.0) or 0.0) / 100.0,
                "oi": pe.get("openInterest", 0) or 0,
                "bidQty": pe.get("bidQty", 0) or 0,
                "ltp": pe.get("lastPrice", 0.0) or 0.0,
            })

    strikes = sorted(set(strikes_all))
    if not strikes:
        return None

    # IV sums (in % for display)
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)
    put_iv_sum = sum((r["iv"] * 100.0) for r in pe_rows)

    # Min positive bid qty per side (used as scaling factor everywhere)
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1

    min_call_q = min_pos(ce_rows)
    min_put_q = min_pos(pe_rows)

    call_value_sum, put_value_sum = 0.0, 0.0
    tot_call_oi, tot_put_oi = 0, 0

    per_strike_gex = {}  # strike -> (call_gex, put_gex, net_gex = call - put)
    per_strike_dex = {}  # strike -> (call_dex, put_dex, net_dex = call + put)

    atm_idx = nearest_index(strikes, spot)
    atm_strike = strikes[atm_idx] if atm_idx is not None else None

    ce_map = {r["strike"]: r for r in ce_rows}
    pe_map = {r["strike"]: r for r in pe_rows}

    for s in strikes:
        ce = ce_map.get(s)
        pe = pe_map.get(s)

        call_dex = put_dex = 0.0
        call_gex = put_gex = 0.0

        if ce and ce["oi"] > 0:
            tot_call_oi += ce["oi"]
            call_value_sum += ce["oi"] * min_call_q * ce["ltp"]
            sigma_c = max(ce["iv"], 1e-6)
            delta_c, gamma_c = bs_delta_gamma(spot, s, r, q, sigma_c, T, "C")
            call_dex = delta_c * ce["oi"] * min_call_q
            call_gex = gamma_c * ce["oi"] * min_call_q * (spot ** 2)

        if pe and pe["oi"] > 0:
            tot_put_oi += pe["oi"]
            put_value_sum += pe["oi"] * min_put_q * pe["ltp"]
            sigma_p = max(pe["iv"], 1e-6)
            delta_p, gamma_p = bs_delta_gamma(spot, s, r, q, sigma_p, T, "P")
            put_dex = delta_p * pe["oi"] * min_put_q  # delta_p negative
            put_gex = gamma_p * pe["oi"] * min_put_q * (spot ** 2)

        per_strike_dex[s] = (call_dex, put_dex, call_dex + put_dex)
        per_strike_gex[s] = (call_gex, put_gex, call_gex - put_gex)

    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi > 0 else None

    def collect_values(rows, mult):
        return [{"strike": r["strike"], "value": r["oi"] * mult * r["ltp"]} for r in rows]

    top_calls = sorted(collect_values(ce_rows, min_call_q), key=lambda x: x["value"], reverse=True)[:2]
    top_puts  = sorted(collect_values(pe_rows, min_put_q),  key=lambda x: x["value"], reverse=True)[:2]

    # Max Pain
    def compute_max_pain():
        all_s = sorted(set([r["strike"] for r in ce_rows] + [r["strike"] for r in pe_rows]))
        call_oi_map = {r["strike"]: r["oi"] for r in ce_rows}
        put_oi_map  = {r["strike"]: r["oi"] for r in pe_rows}
        best, best_pay = None, None
        for s in all_s:
            pay = 0
            for k in all_s:
                if s > k:  # calls ITM
                    pay += call_oi_map.get(k, 0) * (s - k)
                if k > s:  # puts ITM
                    pay += put_oi_map.get(k, 0) * (k - s)
            if best_pay is None or pay < best_pay:
                best, best_pay = s, pay
        return best

    max_pain = compute_max_pain() if (ce_rows or pe_rows) else None

    # IV skew near ATM
    def avg_iv_near_atm(window=1):
        if atm_strike is None:
            return None, None, None
        idx = strikes.index(atm_strike)
        span = strikes[max(0, idx - window): idx + window + 1]
        call_ivs = [ce_map[s]["iv"] * 100.0 for s in span if s in ce_map]  # %
        put_ivs  = [pe_map[s]["iv"] * 100.0 for s in span if s in pe_map]  # %
        if not call_ivs or not put_ivs:
            return None, None, None
        call_iv = sum(call_ivs) / len(call_ivs)
        put_iv  = sum(put_ivs) / len(put_ivs)
        return call_iv, put_iv, (put_iv - call_iv)

    call_iv_atm, put_iv_atm, iv_skew = avg_iv_near_atm(window=1)

    # Totals
    gex_call_total = sum(v[0] for v in per_strike_gex.values())
    gex_put_total  = sum(v[1] for v in per_strike_gex.values())
    gex_net        = gex_call_total - gex_put_total
    dex_net        = sum(v[2] for v in per_strike_dex.values())

    gamma_walls = sorted(
        [{"strike": s, "net_gex": v[2]} for s, v in per_strike_gex.items()],
        key=lambda x: abs(x["net_gex"]), reverse=True
    )[:3]

    delta_walls = sorted(
        [{"strike": s, "net_dex": per_strike_dex[s][2], "call_dex": per_strike_dex[s][0], "put_dex": per_strike_dex[s][1]}
         for s in per_strike_dex],
        key=lambda x: abs(x["net_dex"]), reverse=True
    )[:3]

    return {
        "spot": spot,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum":  put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum":  put_value_sum,
        "total_call_oi": tot_call_oi,
        "total_put_oi":  tot_put_oi,
        "pcr": pcr,
        "max_pain": max_pain,
        "iv_skew": iv_skew,
        "call_iv_atm": call_iv_atm,
        "put_iv_atm": put_iv_atm,
        "atm_strike": atm_strike,
        "gex_call_total": gex_call_total,
        "gex_put_total":  gex_put_total,
        "gex_net": gex_net,
        "dex_net": dex_net,
        "gamma_walls": gamma_walls,
        "delta_walls": delta_walls,
        "top_calls": top_calls,
        "top_puts":  top_puts,
        "min_call_q": min_call_q,
        "min_put_q":  min_put_q,
    }

def term_structure_atm_iv(data: dict, expiries: list, spot: float):
    """
    Compare ATM IV (near) vs next expiry.
    Returns (near_iv, next_iv, ratio, diff) in %
    """
    if not expiries or spot is None:
        return None, None, None, None

    ce_by_exp, pe_by_exp = {}, {}
    rows = (data or {}).get("records", {}).get("data", []) or []
    for item in rows:
        exp = item.get("expiryDate")
        strike = item.get("strikePrice")
        ce, pe = item.get("CE"), item.get("PE")
        ce_by_exp.setdefault(exp, {})
        pe_by_exp.setdefault(exp, {})
        if ce:
            ce_by_exp[exp][strike] = (ce.get("impliedVolatility", 0.0) or 0.0) / 100.0
        if pe:
            pe_by_exp[exp][strike] = (pe.get("impliedVolatility", 0.0) or 0.0) / 100.0

    def atm_iv_for_exp(exp):
        if exp not in ce_by_exp or exp not in pe_by_exp:
            return None
        strikes = sorted(set(list(ce_by_exp[exp].keys()) + list(pe_by_exp[exp].keys())))
        if not strikes:
            return None
        idx = nearest_index(strikes, spot)
        if idx is None:
            return None
        atm_s = strikes[idx]
        c_iv = ce_by_exp[exp].get(atm_s)
        p_iv = pe_by_exp[exp].get(atm_s)
        if c_iv is None or p_iv is None:
            return None
        return (c_iv + p_iv) / 2.0  # decimal

    near = expiries[0]
    near_iv = atm_iv_for_exp(near)
    next_iv = atm_iv_for_exp(expiries[1]) if len(expiries) > 1 else None
    if near_iv is None or next_iv is None:
        return None, None, None, None
    near_iv_pct = near_iv * 100.0
    next_iv_pct = next_iv * 100.0
    ratio = (near_iv / next_iv) if next_iv > 0 else None
    diff = near_iv_pct - next_iv_pct
    return near_iv_pct, next_iv_pct, ratio, diff

# =========================
# Sidebar inputs
# =========================
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
data = fetch_option_chain(symbol)
if not data:
    st.stop()

expiries = data["records"].get("expiryDates", []) or []
if not expiries:
    st.error("No expiries found.")
    st.stop()

expiry = st.sidebar.selectbox("Select Expiry", expiries)
if st.sidebar.button("Clear History"):
    st.session_state["history"] = []

# =========================
# Compute Metrics
# =========================
m = compute_metrics(data, symbol, expiry)
if m is None:
    st.error("Unable to compute metrics for the selected expiry.")
    st.stop()

# Previous snapshot (for diffs & Vanna)
prev = st.session_state.get("history", [])[-1] if st.session_state.get("history") else {}
def diff(curr, prevv):
    return "–" if (prevv is None) else f"{curr - prevv:+.2f}"

# =========================
# Headline Metrics
# =========================
st.subheader("Market Snapshot")
c0, c1, c2 = st.columns(3)
with c0:
    st.metric("Spot Price", f"{m['spot']:.2f}" if m['spot'] else "-")
with c1:
    st.metric("Call IV Sum", f"{m['call_iv_sum']:.2f}", diff(m['call_iv_sum'], prev.get("Call IV Sum")))
    st.metric("Call Value", format_inr(m['call_value_sum']), diff(m['call_value_sum'], prev.get("Call Value")))
with c2:
    st.metric("Put IV Sum", f"{m['put_iv_sum']:.2f}", diff(m['put_iv_sum'], prev.get("Put IV Sum")))
    st.metric("Put Value", format_inr(m['put_value_sum']), diff(m['put_value_sum'], prev.get("Put Value")))

# =========================
# Institutional Metrics
# =========================
st.subheader("Institutional Metrics")

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("PCR", f"{m['pcr']:.2f}" if m["pcr"] is not None else "-")
    if m["pcr"] is not None:
        st.caption("Action: Buy (bullish)" if m["pcr"] > 1 else "Action: Sell (bearish)")

with d2:
    st.metric("Max Pain", f"{m['max_pain']}" if m["max_pain"] else "-")
    if m["max_pain"]:
        st.caption("Action: Buy (pull up)" if m["spot"] < m["max_pain"] else "Action: Sell (pull down)")

with d3:
    st.metric("IV Skew (Put - Call) %", f"{m['iv_skew']:.2f}" if m["iv_skew"] is not None else "-")
    if m["iv_skew"] is not None:
        st.caption("Action: Sell (downside fear)" if m["iv_skew"] > 0 else "Action: Buy (upside demand)")

with d4:
    delta_call = m["total_call_oi"] - prev.get("Total Call OI", 0) if prev else 0
    delta_put  = m["total_put_oi"] - prev.get("Total Put OI", 0) if prev else 0
    st.metric("OI Flow Δ", f"C:{delta_call} / P:{delta_put}")
    if (delta_call != 0) or (delta_put != 0):
        st.caption("Action: Sell (call writing)" if delta_call > delta_put else "Action: Buy (put writing)")
    else:
        st.caption("Action: Neutral (no fresh positioning)")

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.metric("GEX Net (scaled)", format_inr(m["gex_net"]))
    st.caption("Action: Caution (range)" if m["gex_net"] > 0 else "Action: Trend (follow breakout)")

with g2:
    if m["gamma_walls"]:
        walls = ", ".join([f"{w['strike']}" for w in m["gamma_walls"]])
        st.metric("Gamma Walls (Top)", walls)
        st.caption("Action: Caution near walls (pin/magnet)")
    else:
        st.metric("Gamma Walls (Top)", "-")

with g3:
    st.metric("DEX Net (scaled)", format_inr(m["dex_net"]))
    st.caption("Action: Up-bias if +ve; Down-bias if -ve")

with g4:
    if m["delta_walls"]:
        notes = []
        for w in m["delta_walls"]:
            dom = "Resist" if abs(w["call_dex"]) >= abs(w["put_dex"]) else "Support"
            notes.append(f"{w['strike']} ({dom})")
        st.metric("Delta Walls (Top)", ", ".join(notes))
        st.caption("Action: Expect stall at walls (hedging)")
    else:
        st.metric("Delta Walls (Top)", "-")

# IV Term Structure
near_iv, next_iv, ts_ratio, ts_diff = term_structure_atm_iv(data, expiries, m["spot"])
t1, t2 = st.columns(2)
with t1:
    st.metric("Term Struct ATM IV (Near/Next) %", f"{near_iv:.2f}/{next_iv:.2f}" if (near_iv and next_iv) else "-")
with t2:
    if ts_ratio is not None:
        st.metric("IV Ratio (Near/Next)", f"{ts_ratio:.2f}")
        st.caption("Action: Caution (event risk)" if ts_ratio > 1.05 else "Action: IV calm (premium selling ok)")
    else:
        st.metric("IV Ratio (Near/Next)", "-")

# Vanna & Charm simple hints
v1, v2 = st.columns(2)
curr_atm_iv = None
if m["call_iv_atm"] is not None and m["put_iv_atm"] is not None:
    curr_atm_iv = (m["call_iv_atm"] + m["put_iv_atm"]) / 2.0

with v1:
    prev_atm_iv = prev.get("ATM IV (near)")
    if curr_atm_iv is not None:
        iv_change = None if prev_atm_iv is None else (curr_atm_iv - prev_atm_iv)
        st.metric("Vanna hint (ATM IV %)", f"{curr_atm_iv:.2f}" if curr_atm_iv is not None else "-")
        if iv_change is not None:
            st.caption("Action: Buy (IV↑ → hedge buying)" if iv_change > 0.3 else
                       "Action: Sell (IV↓ → hedge selling)" if iv_change < -0.3 else
                       "Action: Neutral (flat IV)")
        else:
            st.caption("Action: Watch (first reading)")
    else:
        st.metric("Vanna hint (ATM IV %)", "-")

with v2:
    if m["atm_strike"] is not None and m["spot"] is not None:
        charm_bias = "Buy (bullish drift)" if m["spot"] >= m["atm_strike"] else "Sell (bearish drift)"
        st.metric("Charm hint (moneyness)", f"Spot/ATM: {m['spot']:.0f}/{m['atm_strike']}")
        st.caption(f"Action: {charm_bias}")
    else:
        st.metric("Charm hint (moneyness)", "-")

# =========================
# Final Comment (Algo)
# =========================
def final_signal(m, prev, curr_atm_iv, ts_ratio):
    score = 0.0
    notes = []

    # PCR
    if m["pcr"] is not None:
        if m["pcr"] > 1.05:
            score += 1; notes.append("PCR>1.05 bullish")
        elif m["pcr"] < 0.95:
            score -= 1; notes.append("PCR<0.95 bearish")

    # Max Pain pull
    if m["max_pain"] and m["spot"]:
        diffp = (m["spot"] - m["max_pain"]) / m["spot"]
        if diffp < -0.005:
            score += 1; notes.append("Spot below MaxPain (pull up)")
        elif diffp > 0.005:
            score -= 1; notes.append("Spot above MaxPain (pull down)")

    # IV Skew
    if m["iv_skew"] is not None:
        if m["iv_skew"] > 0.5:
            score -= 1; notes.append("Put IV>Call IV (downside fear)")
        elif m["iv_skew"] < -0.5:
            score += 1; notes.append("Call IV>Put IV (upside demand)")

    # OI Flow Δ
    prev_call_oi = prev.get("Total Call OI", 0) if prev else 0
    prev_put_oi  = prev.get("Total Put OI", 0) if prev else 0
    d_call = m["total_call_oi"] - prev_call_oi
    d_put  = m["total_put_oi"] - prev_put_oi
    if (d_call != 0) or (d_put != 0):
        if d_put > d_call * 1.1:
            score += 1; notes.append("Put OI rising faster")
        elif d_call > d_put * 1.1:
            score -= 1; notes.append("Call OI rising faster")

    # DEX net
    if m["dex_net"] is not None:
        if m["dex_net"] > 0:
            score += 0.5; notes.append("DEX>0 up-bias")
        elif m["dex_net"] < 0:
            score -= 0.5; notes.append("DEX<0 down-bias")

    # Vanna: ATM IV change vs prev
    prev_atm_iv = prev.get("ATM IV (near)")
    if curr_atm_iv is not None and prev_atm_iv is not None:
        dv = curr_atm_iv - prev_atm_iv
        if dv > 0.3:
            score += 0.5; notes.append("IV rising (Vanna buy)")
        elif dv < -0.3:
            score -= 0.5; notes.append("IV falling (Vanna sell)")

    # Charm: moneyness drift
    if m["atm_strike"] is not None and m["spot"] is not None:
        if m["spot"] >= m["atm_strike"]:
            score += 0.25; notes.append("Charm bullish drift")
        else:
            score -= 0.25; notes.append("Charm bearish drift")

    # Term structure guard: strong near-term event risk -> BeSeated
    if ts_ratio is not None and ts_ratio > 1.08:
        return "BeSeated", "Near-term IV >> next (event risk). " + "; ".join(notes)

    # GEX regime adjustment
    if m["gex_net"] is not None:
        if m["gex_net"] < 0:
            if score > 0: score += 0.5; notes.append("GEX<0 amplifies up")
            elif score < 0: score -= 0.5; notes.append("GEX<0 amplifies down")
        elif m["gex_net"] > 0:
            if score > 0: score -= 0.25; notes.append("GEX>0 dampens up")
            elif score < 0: score += 0.25; notes.append("GEX>0 dampens down")

    # Final decision
    if score >= 2:
        return "BuyCall", "; ".join(notes)
    elif score <= -2:
        return "BuyPut", "; ".join(notes)
    else:
        return "BeSeated", "; ".join(notes)

signal, rationale = final_signal(m, prev, curr_atm_iv, ts_ratio)

st.subheader("Final Comment")
st.metric("Decision", signal)
st.caption(rationale if rationale else "—")

# =========================
# Top Strikes by your Value
# =========================
st.subheader(f"Top Strikes by Value (Expiry {expiry})")
cc, pp = st.columns(2)
with cc:
    st.write("Top Calls")
    for r in m["top_calls"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")
with pp:
    st.write("Top Puts")
    for r in m["top_puts"]:
        st.write(f"Strike {r['strike']}: {format_inr(r['value'])}")

# =========================
# Snapshot History (auto + manual)
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

if refresh_btn or auto_snapshot_due():
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["history"].append({
        "Time": ts,
        "Symbol": symbol,
        "Expiry": expiry,
        "Spot": m["spot"],
        "Call IV Sum": m["call_iv_sum"],
        "Put IV Sum": m["put_iv_sum"],
        "Call Value": m["call_value_sum"],
        "Put Value": m["put_value_sum"],
        "PCR": m["pcr"],
        "Max Pain": m["max_pain"],
        "IV Skew": m["iv_skew"],
        "ATM IV (near)": curr_atm_iv if curr_atm_iv is not None else None,
        "GEX Net": m["gex_net"],
        "DEX Net": m["dex_net"],
        "Total Call OI": m["total_call_oi"],
        "Total Put OI": m["total_put_oi"],
        "Decision": signal
    })

st.subheader("Snapshot History")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "history.csv")
else:
    st.info("Snapshots will appear here every ~10 minutes or on Refresh Now.")
