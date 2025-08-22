# --------------------------
# Metrics
# --------------------------
metrics = compute_metrics(chain_data, expiry)

# Compare with last snapshot if available
prev_call_val, prev_put_val, prev_call_iv, prev_put_iv = None, None, None, None
if "history" in st.session_state and st.session_state["history"]:
    prev = st.session_state["history"][-1]
    prev_call_val = prev["Call Value"]
    prev_put_val = prev["Put Value"]
    prev_call_iv = prev["Call IV Sum"]
    prev_put_iv = prev["Put IV Sum"]

def diff_text(curr, prev, fmt="num"):
    if prev is None:
        return "–"
    diff = curr - prev
    if diff > 0:
        return f"▲ {format_inr(diff) if fmt=='inr' else round(diff,2)}"
    elif diff < 0:
        return f"▼ {format_inr(abs(diff)) if fmt=='inr' else round(diff,2)}"
    else:
        return "No change"

c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Spot Price", f"{metrics['spot']:.2f}" if metrics['spot'] else "-")
c1.metric(
    "Call IV Sum",
    f"{metrics['call_iv_sum']:.2f}",
    diff_text(metrics["call_iv_sum"], prev_call_iv, fmt="num")
)
c2.metric(
    "Put IV Sum",
    f"{metrics['put_iv_sum']:.2f}",
    diff_text(metrics["put_iv_sum"], prev_put_iv, fmt="num")
)
c3.metric(
    "Call Value",
    format_inr(metrics["call_value_sum"]),
    diff_text(metrics["call_value_sum"], prev_call_val, fmt="inr")
)
c4.metric(
    "Put Value",
    format_inr(metrics["put_value_sum"]),
    diff_text(metrics["put_value_sum"], prev_put_val, fmt="inr")
)
