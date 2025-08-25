# (imports and NSE client remain unchanged above this point...)

# =========================
# Compute Metrics
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

    # IV sums
    call_iv_sum = sum((r["iv"] * 100.0) for r in ce_rows)
    put_iv_sum  = sum((r["iv"] * 100.0) for r in pe_rows)

    # Min bid qty scaling
    def min_pos(rows):
        vals = [r["bidQty"] for r in rows if r["bidQty"] > 0]
        return min(vals) if vals else 1

    min_call_q = min_pos(ce_rows)
    min_put_q  = min_pos(pe_rows)

    tot_call_oi, tot_put_oi = 0, 0
    call_value_sum, put_value_sum = 0.0, 0.0
    dex_net, gex_net = 0.0, 0.0

    atm_idx = nearest_index(strikes, spot)
    atm_strike = strikes[atm_idx] if atm_idx is not None else None
    ce_map = {r["strike"]: r for r in ce_rows}
    pe_map = {r["strike"]: r for r in pe_rows}

    for s in strikes:
        ce = ce_map.get(s)
        pe = pe_map.get(s)
        if ce and ce["oi"] > 0:
            tot_call_oi += ce["oi"]
            call_value_sum += ce["oi"] * min_call_q * ce["ltp"]
            sigma_c = max(ce["iv"], 1e-6)
            d_c, g_c = bs_delta_gamma(spot, s, r, q, sigma_c, T, "C")
            dex_net += d_c * ce["oi"] * min_call_q
            gex_net += g_c * ce["oi"] * min_call_q * (spot**2)
        if pe and pe["oi"] > 0:
            tot_put_oi += pe["oi"]
            put_value_sum += pe["oi"] * min_put_q * pe["ltp"]
            sigma_p = max(pe["iv"], 1e-6)
            d_p, g_p = bs_delta_gamma(spot, s, r, q, sigma_p, T, "P")
            dex_net += d_p * pe["oi"] * min_put_q
            gex_net -= g_p * pe["oi"] * min_put_q * (spot**2)

    pcr = (tot_put_oi / tot_call_oi) if tot_call_oi > 0 else None

    # IV Skew near ATM
    def avg_iv_near_atm(window=1):
        if atm_strike is None:
            return None, None, None
        idx = strikes.index(atm_strike)
        span = strikes[max(0, idx - window): idx + window + 1]
        call_ivs = [ce_map[s]["iv"] * 100.0 for s in span if s in ce_map]
        put_ivs  = [pe_map[s]["iv"] * 100.0 for s in span if s in pe_map]
        if not call_ivs or not put_ivs:
            return None, None, None
        return sum(call_ivs)/len(call_ivs), sum(put_ivs)/len(put_ivs), (sum(put_ivs)/len(put_ivs) - sum(call_ivs)/len(call_ivs))

    call_iv_atm, put_iv_atm, iv_skew = avg_iv_near_atm()
    atm_iv_now = None
    if call_iv_atm and put_iv_atm:
        atm_iv_now = (call_iv_atm + put_iv_atm)/2.0

    # Max Pain
    def compute_max_pain():
        all_s = sorted(set([r["strike"] for r in ce_rows] + [r["strike"] for r in pe_rows]))
        call_oi_map = {r["strike"]: r["oi"] for r in ce_rows}
        put_oi_map  = {r["strike"]: r["oi"] for r in pe_rows}
        best, best_pay = None, None
        for s in all_s:
            pay = 0
            for k in all_s:
                if s > k:
                    pay += call_oi_map.get(k, 0) * (s - k)
                if k > s:
                    pay += put_oi_map.get(k, 0) * (k - s)
            if best_pay is None or pay < best_pay:
                best, best_pay = s, pay
        return best

    max_pain = compute_max_pain() if (ce_rows or pe_rows) else None

    return {
        "spot": spot,
        "expiry": expiry,
        "call_iv_sum": call_iv_sum,
        "put_iv_sum": put_iv_sum,
        "call_value_sum": call_value_sum,
        "put_value_sum": put_value_sum,
        "total_call_oi": tot_call_oi,
        "total_put_oi": tot_put_oi,
        "pcr": pcr,
        "iv_skew": iv_skew,
        "atm_iv_now": atm_iv_now,
        "atm_strike": atm_strike,
        "max_pain": max_pain,
        "dex_net": dex_net,
        "gex_net": gex_net,
        "min_call_q": min_call_q,
        "min_put_q": min_put_q,
    }

# =========================
# Final Comment
# =========================
def final_comment(m, prev):
    pos, neg, mods = [], [], []

    # PCR
    if m["pcr"]:
        if m["pcr"] > 1.1: pos.append("put buildup supports upside")
        elif m["pcr"] < 0.9: neg.append("call positioning adds resistance")

    # Max Pain pull
    if m["max_pain"] and m["spot"]:
        pull = (m["max_pain"] - m["spot"]) / m["spot"]
        if pull > 0.005: pos.append("expiry pull favors upticks")
        elif pull < -0.005: neg.append("expiry pull favors downticks")

    # IV Skew
    if m["iv_skew"]:
        if m["iv_skew"] > 0.5: neg.append("downside skew signals caution")
        elif m["iv_skew"] < -0.5: pos.append("upside skew shows appetite")

    # OI Flow Δ
    if prev:
        d_call = m["total_call_oi"] - (prev.get("Total Call OI", 0) or 0)
        d_put  = m["total_put_oi"] - (prev.get("Total Put OI", 0) or 0)
        if d_put > d_call * 1.1: pos.append("fresh put build-up aids bids")
        elif d_call > d_put * 1.1: neg.append("fresh call build-up caps rallies")

    # DEX
    if m["dex_net"] > 0: pos.append("dealer hedging adds support")
    elif m["dex_net"] < 0: neg.append("dealer hedging adds pressure")

    # Vanna (ATM IV change)
    prev_iv = prev.get("ATM IV (near)")
    if m["atm_iv_now"] and prev_iv:
        dv = m["atm_iv_now"] - prev_iv
        if dv > 0.3: mods.append("rising IV fuels hedge demand")
        elif dv < -0.3: mods.append("IV easing reduces hedge demand")

    # Charm
    if m["atm_strike"] and m["spot"]:
        if m["spot"] >= m["atm_strike"]: mods.append("time-decay drift leans up")
        else: mods.append("time-decay drift leans down")

    # GEX
    if m["gex_net"] > 0: mods.append("positive gamma keeps moves steady")
    elif m["gex_net"] < 0: mods.append("negative gamma can amplify swings")

    # Build summary
    if len(pos) > len(neg)+1: first = "Bullish bias — " + ", ".join(pos[:2])
    elif len(neg) > len(pos)+1: first = "Bearish bias — " + ", ".join(neg[:2])
    else:
        mix = pos[:1] + neg[:1]
        first = "Flows mixed — " + ", ".join(mix) if mix else "Flows mixed — positioning balanced"

    second = ". " + "; ".join(mods[:2]) if mods else ""
    return (first + second).strip()
