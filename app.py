"""
app.py — Bayesian A/B Testing Dashboard
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from bayesian_ab import (
    BetaPosterior, update_posterior,
    prob_b_beats_a_mc, expected_loss, relative_uplift,
    simulate_experiment, frequentist_test, required_sample_size,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bayesian A/B Tester", page_icon="⚗️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #1a1625;
    color: #e2e0e7;
  }
  h1 { font-family: 'Lora', serif; font-weight: 600; }
  h2, h3 { font-family: 'Inter', sans-serif; font-weight: 500; }

  .card {
    background: #241f32;
    border: 1px solid #352e47;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .card-label { color: #6b6480; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
  .card-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 500; }
  .card-sub   { color: #524b68; font-size: 11px; margin-top: 4px; }

  .verdict-b {
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.4);
    border-left: 4px solid #8b5cf6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 10px 0; font-size: 15px;
  }
  .verdict-a {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-left: 4px solid #fbbf24;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 10px 0; font-size: 15px;
  }
  .explainer {
    background: #241f32;
    border: 1px solid #352e47;
    border-radius: 10px;
    padding: 16px 20px; margin: 10px 0; font-size: 14px;
    line-height: 1.7; color: #a89ec0;
  }
  .explainer b { color: #e2e0e7; }
  .formula {
    font-family: 'JetBrains Mono', monospace;
    background: #13101e;
    border-radius: 6px;
    padding: 10px 14px; margin: 8px 0;
    font-size: 13px; color: #c4b5fd;
  }

  div[data-testid="stSidebar"] { background: #13101e; border-right: 1px solid #241f32; }
</style>
""", unsafe_allow_html=True)

BG   = "#1a1625"
CARD = "#241f32"
GRID = "#2d2640"
FONT = dict(color="#6b6480", family="Inter")
COL_A = "#fbbf24"   # amber for control
COL_B = "#8b5cf6"   # purple for variant

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ Bayesian\nA/B Tester")
    st.markdown("---")
    page = st.radio("", [
        "📖 The Method",
        "🧪 Run an Experiment",
        "📈 Live Simulation",
        "⚖️ Bayes vs Frequentist",
        "📐 Sample Size Calculator",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#524b68;line-height:1.6'>"
        "Beta-Binomial conjugate model · "
        "Monte Carlo posterior sampling · "
        "Expected loss decision criterion"
        "</div>", unsafe_allow_html=True
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ Bayesian A/B Testing")
st.markdown("*From p-values to posterior probabilities — a principled framework for conversion experiments*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: THE METHOD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📖 The Method":

    st.markdown("### Why Bayesian?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="explainer">
        <b>The frequentist approach gives you a p-value.</b><br><br>
        A p-value is the probability of seeing data <i>this extreme or more extreme</i>,
        assuming the null hypothesis (no difference) is true.<br><br>
        This is almost never what you actually want to know.
        What you want is: <b>"what is the probability that B is better than A?"</b><br><br>
        A p-value cannot answer this question. It is routinely misinterpreted as if it can.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="explainer">
        <b>The Bayesian approach gives you a posterior distribution.</b><br><br>
        We model the true conversion rate θ as a random variable with a prior belief,
        then update that belief as data arrives. The result is a full probability
        distribution over possible values of θ — from which we can directly compute:<br><br>
        • P(B > A) — probability variant B is better<br>
        • Credible intervals — where the true rate probably lies<br>
        • Expected loss — cost of being wrong about our decision
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### The Model")

    st.markdown("""
    <div class="explainer">
    We model conversion as a sequence of Bernoulli trials — each visitor either converts (1) or doesn't (0).
    The unknown true conversion rate θ follows a <b>Beta distribution</b>:<br><br>
    </div>
    <div class="formula">Prior:     θ ~ Beta(α₀, β₀)      ← our belief before data
    Likelihood: data | θ ~ Binomial(n, θ)   ← how data is generated
    Posterior:  θ | data ~ Beta(α₀ + k, β₀ + n - k)  ← updated belief</div>
    <div class="explainer">
    where k = conversions, n = visitors. This is <b>Beta-Binomial conjugacy</b> — the posterior
    is analytically tractable (no sampling required for the update step).<br><br>
    We use a <b>uniform prior</b> Beta(1,1) — expressing no prior belief about the conversion rate.
    As data accumulates, the posterior tightens around the true value.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Visual intuition: how priors update")

    # Show prior → posterior with different data amounts
    x = np.linspace(0, 0.3, 500)

    scenarios = [
        ("Prior (no data)",      1,   1,   "#524b68"),
        ("10 visitors, 1 conv",  2,  10,   "#6b6480"),
        ("100 visitors, 8 conv", 9,  93,   COL_A),
        ("500 visitors, 42 conv",43, 459,  COL_B),
    ]

    fig = go.Figure()
    for label, a, b, color in scenarios:
        post = BetaPosterior(a, b)
        fig.add_trace(go.Scatter(
            x=x, y=post.pdf(x),
            mode="lines", name=label,
            line=dict(color=color, width=2.5),
            fill="tozeroy" if label == "500 visitors, 42 conv" else None,
            fillcolor="rgba(139,92,246,0.08)",
        ))

    fig.update_layout(
        title="Posterior belief tightens as data arrives",
        xaxis=dict(title="Conversion Rate θ", gridcolor=GRID, tickformat=".0%"),
        yaxis=dict(title="Probability Density", gridcolor=GRID),
        plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=40,b=10), height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The uniform prior (flat line) gives way to an increasingly peaked distribution "
        "as more data arrives. With 500 visitors, we're quite confident the true rate "
        "is near 8.4% — the posterior is tight around that value."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RUN AN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Run an Experiment":
    st.markdown("### Analyse an Experiment")
    st.caption("Enter observed data and get full Bayesian inference results.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Control (A)")
        v_a = st.number_input("Visitors A", 100, 1_000_000, 5000, 100)
        c_a = st.number_input("Conversions A", 0, int(v_a), 250, 10)
    with col2:
        st.markdown("#### Variant (B)")
        v_b = st.number_input("Visitors B", 100, 1_000_000, 5000, 100)
        c_b = st.number_input("Conversions B", 0, int(v_b), 310, 10)

    col3, col4 = st.columns(2)
    with col3:
        prior_a = st.slider("Prior α (prior successes)", 1.0, 20.0, 1.0, 0.5,
                            help="1 = uniform/uninformative prior")
        prior_b = st.slider("Prior β (prior failures)",  1.0, 20.0, 1.0, 0.5)
    with col4:
        decision_threshold = st.slider("Decision Threshold", 0.80, 0.999, 0.95, 0.005,
                                       format="%.3f")

    # Compute
    post_a = update_posterior(prior_a, prior_b, c_a, v_a)
    post_b = update_posterior(prior_a, prior_b, c_b, v_b)
    post_a.name = "Control (A)"
    post_b.name = "Variant (B)"

    p_b_wins = prob_b_beats_a_mc(post_a, post_b)
    loss     = expected_loss(post_a, post_b)
    uplift   = relative_uplift(post_a, post_b)
    ci_a     = post_a.credible_interval()
    ci_b     = post_b.credible_interval()
    freq     = frequentist_test(c_a, v_a, c_b, v_b)

    # Decision
    if p_b_wins >= decision_threshold:
        verdict_class, verdict_text = "verdict-b", f"✅ Ship variant B — P(B > A) = {p_b_wins:.1%}"
    elif (1 - p_b_wins) >= decision_threshold:
        verdict_class, verdict_text = "verdict-a", f"⚠️ Keep control A — P(A > B) = {1-p_b_wins:.1%}"
    else:
        verdict_class, verdict_text = "verdict-a", f"⏳ Keep running — P(B > A) = {p_b_wins:.1%}, not yet decisive"

    st.markdown(f'<div class="{verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, sub in [
        (c1, "Rate A",    f"{c_a/v_a:.2%}", f"{c_a:,} / {v_a:,}"),
        (c2, "Rate B",    f"{c_b/v_b:.2%}", f"{c_b:,} / {v_b:,}"),
        (c3, "P(B > A)",  f"{p_b_wins:.1%}", "Bayesian"),
        (c4, "Median Uplift", f"{uplift['p50']:.1%}", f"95% CI [{uplift['p2_5']:.1%}, {uplift['p97_5']:.1%}]"),
        (c5, "Expected Loss", f"{min(loss['loss_if_choose_a'], loss['loss_if_choose_b']):.4f}",
             "Min of choose-A / choose-B"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-label">{label}</div>
              <div class="card-value" style="font-size:20px">{val}</div>
              <div class="card-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Posterior distributions
    col_post, col_uplift = st.columns(2)

    with col_post:
        st.markdown("#### Posterior Distributions")
        lo = min(post_a.mean, post_b.mean) * 0.6
        hi = max(post_a.mean, post_b.mean) * 1.4
        x  = np.linspace(lo, hi, 600)

        fig = go.Figure()
        for post, color, fill_c in [
            (post_a, COL_A, "rgba(251,191,36,0.12)"),
            (post_b, COL_B, "rgba(139,92,246,0.12)"),
        ]:
            ci = post.credible_interval()
            fig.add_trace(go.Scatter(
                x=x, y=post.pdf(x), mode="lines",
                name=f"{post.name} (mean={post.mean:.2%})",
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor=fill_c,
            ))
            # CI shading
            x_ci = np.linspace(ci[0], ci[1], 200)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_ci, x_ci[::-1]]),
                y=np.concatenate([post.pdf(x_ci), np.zeros(200)]),
                fill="toself", fillcolor=fill_c.replace("0.12","0.25"),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))

        fig.update_layout(
            xaxis=dict(title="Conversion Rate", gridcolor=GRID, tickformat=".1%"),
            yaxis=dict(title="Density", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=20,b=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded regions = 95% credible intervals. Overlap indicates uncertainty.")

    with col_uplift:
        st.markdown("#### Relative Uplift Distribution")
        samples = np.array(uplift["samples"])
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=samples, nbinsx=80,
            marker_color=COL_B, opacity=0.75,
            histnorm="probability density", name="(B-A)/A",
        ))
        fig2.add_vline(x=0, line_color="#ef4444", line_dash="dash",
                       annotation_text="No difference", annotation_position="top left")
        fig2.add_vline(x=uplift["p50"], line_color=COL_B,
                       annotation_text=f"Median {uplift['p50']:.1%}",
                       annotation_position="top right")
        fig2.update_layout(
            xaxis=dict(title="Relative Uplift (B vs A)", gridcolor=GRID, tickformat=".0%"),
            yaxis=dict(title="Density", gridcolor=GRID),
            plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
            margin=dict(l=10,r=10,t=20,b=10), height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"P(uplift > 0) = {(samples > 0).mean():.1%}. "
                   f"95% CI: [{uplift['p2_5']:.1%}, {uplift['p97_5']:.1%}]")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Live Simulation":
    st.markdown("### Day-by-Day Experiment Simulation")
    st.caption("See how the Bayesian inference evolves as data accumulates over the experiment window.")

    col1, col2, col3 = st.columns(3)
    with col1:
        true_a = st.slider("True Rate A (%)", 1.0, 20.0, 5.0, 0.5) / 100
        true_b = st.slider("True Rate B (%)", 1.0, 20.0, 6.5, 0.5) / 100
    with col2:
        daily  = st.slider("Daily Visitors (total)", 100, 5000, 500, 50)
        n_days = st.slider("Experiment Duration (days)", 7, 60, 30)
    with col3:
        thresh = st.slider("Decision Threshold", 0.80, 0.999, 0.95, 0.005, format="%.3f")
        seed   = st.number_input("Random Seed", 1, 999, 42)

    @st.cache_data(show_spinner="Simulating…")
    def run_sim(ta, tb, dv, nd, th, sd):
        return simulate_experiment(ta, tb, dv, nd, decision_threshold=th, seed=sd)

    result = run_sim(true_a, true_b, daily, n_days, thresh, int(seed))
    h      = pd.DataFrame(result["history"])
    decision_day = h[h["decision"].notna()].index[0] + 1 if h["decision"].notna().any() else None

    # P(B > A) over time
    fig = go.Figure()
    fig.add_hline(y=thresh, line_dash="dash", line_color=COL_B,
                  annotation_text=f"Decision threshold {thresh:.0%}")
    fig.add_hline(y=1-thresh, line_dash="dash", line_color=COL_A,
                  annotation_text=f"A wins threshold {1-thresh:.0%}",
                  annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=h["day"], y=h["p_b_wins"],
        mode="lines", name="P(B > A)",
        line=dict(color=COL_B, width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
    ))
    if decision_day:
        fig.add_vline(x=decision_day, line_color="#34d399", line_dash="dot",
                      annotation_text=f"Decision: day {decision_day}",
                      annotation_position="top right")
    fig.update_layout(
        title="P(B > A) as experiment runs",
        xaxis=dict(title="Day", gridcolor=GRID),
        yaxis=dict(title="P(B > A)", gridcolor=GRID, tickformat=".0%", range=[0,1]),
        plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
        margin=dict(l=10,r=10,t=40,b=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Observed rates + credible intervals over time
    fig2 = go.Figure()
    fig2.add_hline(y=true_a, line_dash="dot", line_color=COL_A, opacity=0.4,
                   annotation_text=f"True A={true_a:.1%}", annotation_position="left")
    fig2.add_hline(y=true_b, line_dash="dot", line_color=COL_B, opacity=0.4,
                   annotation_text=f"True B={true_b:.1%}", annotation_position="left")

    for label, mean_col, lo_col, hi_col, color in [
        ("Control A", "post_a_mean", "ci_a_lo", "ci_a_hi", COL_A),
        ("Variant B", "post_b_mean", "ci_b_lo", "ci_b_hi", COL_B),
    ]:
        fig2.add_trace(go.Scatter(
            x=pd.concat([h["day"], h["day"][::-1]]),
            y=pd.concat([h[hi_col], h[lo_col][::-1]]),
            fill="toself", fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.12)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            x=h["day"], y=h[mean_col],
            mode="lines", name=label,
            line=dict(color=color, width=2),
        ))

    fig2.update_layout(
        title="Posterior mean + 95% credible interval per variant",
        xaxis=dict(title="Day", gridcolor=GRID),
        yaxis=dict(title="Conversion Rate", gridcolor=GRID, tickformat=".1%"),
        plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=40,b=10), height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Decision summary
    c1,c2,c3 = st.columns(3)
    final = h.iloc[-1]
    for col, label, val in [
        (c1, "Final Decision",      result["decision"]),
        (c2, "Final P(B > A)",      f"{final['p_b_wins']:.1%}"),
        (c3, "Decision Reached Day", str(decision_day) if decision_day else "Not reached"),
    ]:
        with col:
            color = COL_B if "B wins" in str(val) else (COL_A if "A wins" in str(val) else "#94a3b8")
            st.markdown(f"""
            <div class="card">
              <div class="card-label">{label}</div>
              <div class="card-value" style="color:{color};font-size:18px">{val}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: BAYES VS FREQUENTIST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Bayes vs Frequentist":
    st.markdown("### Bayesian vs Frequentist: Side-by-Side")
    st.caption("Same data, same question — two very different answers.")

    col1, col2 = st.columns(2)
    with col1:
        v_a2 = st.number_input("Visitors A", 100, 100_000, 2000, 100, key="v_a2")
        c_a2 = st.number_input("Conversions A", 0, int(v_a2), 100, 5, key="c_a2")
    with col2:
        v_b2 = st.number_input("Visitors B", 100, 100_000, 2000, 100, key="v_b2")
        c_b2 = st.number_input("Conversions B", 0, int(v_b2), 120, 5, key="c_b2")

    post_a2 = update_posterior(1, 1, c_a2, v_a2)
    post_b2 = update_posterior(1, 1, c_b2, v_b2)
    p_b2    = prob_b_beats_a_mc(post_a2, post_b2)
    loss2   = expected_loss(post_a2, post_b2)
    up2     = relative_uplift(post_a2, post_b2)
    freq2   = frequentist_test(c_a2, v_a2, c_b2, v_b2)

    col_bay, col_freq = st.columns(2)

    with col_bay:
        st.markdown(f"#### 🔵 Bayesian Result")
        for label, val, note in [
            ("P(B > A)",             f"{p_b2:.1%}",          "Direct probability B is better"),
            ("Median relative uplift",f"{up2['p50']:.1%}",   f"95% CI [{up2['p2_5']:.1%}, {up2['p97_5']:.1%}]"),
            ("Expected loss (choose B)", f"{loss2['loss_if_choose_b']:.5f}", "Avg loss if we're wrong"),
            ("Decision",             "Ship B" if p_b2>0.95 else "Keep running", "At 95% threshold"),
        ]:
            st.markdown(f"""
            <div class="card" style="text-align:left;margin-bottom:8px">
              <div class="card-label">{label}</div>
              <div class="card-value" style="font-size:18px;color:{COL_B}">{val}</div>
              <div class="card-sub">{note}</div>
            </div>""", unsafe_allow_html=True)

    with col_freq:
        st.markdown(f"#### 🟡 Frequentist Result")
        sig = freq2["significant"]
        for label, val, note in [
            ("p-value",            f"{freq2['p_value']:.4f}", "P(data this extreme | H₀ true)"),
            ("Significant?",       "Yes ✓" if sig else "No ✗", "At α=0.05"),
            ("95% CI on difference", f"[{freq2['diff_lo']:.3%}, {freq2['diff_hi']:.3%}]", "Absolute rate difference"),
            ("What it tells you",  "Reject H₀" if sig else "Fail to reject H₀", "Not: P(B > A)"),
        ]:
            st.markdown(f"""
            <div class="card" style="text-align:left;margin-bottom:8px">
              <div class="card-label">{label}</div>
              <div class="card-value" style="font-size:18px;color:{COL_A}">{val}</div>
              <div class="card-sub">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="explainer">
    <b>The key difference:</b> The frequentist p-value tells you about the data given a hypothesis.
    The Bayesian posterior tells you about the hypothesis given the data — which is what you
    actually care about.<br><br>
    A p-value of 0.03 does NOT mean "there's a 97% chance B is better."
    It means "if the null were true, we'd see data this extreme only 3% of the time."
    These are very different statements.<br><br>
    <b>When frequentist wins:</b> Pre-registered trials with fixed sample sizes, regulatory
    environments requiring p-values, situations where you want strong type-I error control.<br><br>
    <b>When Bayesian wins:</b> Sequential testing (stopping early), incorporating prior knowledge,
    communicating results to non-technical stakeholders, making decisions under uncertainty.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: SAMPLE SIZE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Sample Size Calculator":
    st.markdown("### Sample Size & Power Calculator")
    st.caption(
        "How long do you need to run the experiment to reliably detect the effect you care about? "
        "Uses the frequentist formula — Bayesian planning is more complex but this gives a good baseline."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        baseline = st.slider("Baseline conversion rate (%)", 1.0, 30.0, 5.0, 0.5) / 100
        daily_v  = st.number_input("Daily visitors (total)", 100, 100_000, 1000, 100)
    with col2:
        mde_pct  = st.slider("Min detectable effect (%)", 5, 100, 20, 5,
                              help="Relative lift you want to be able to detect. 20% means: if true rate is 5%, detect lifts to 6%.")
        alpha    = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
    with col3:
        power    = st.select_slider("Power (1-β)", [0.70, 0.80, 0.90], value=0.80)

    n_per_variant = required_sample_size(baseline, mde_pct/100, alpha, power)
    n_total       = n_per_variant * 2
    days_needed   = int(np.ceil(n_total / daily_v))

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Per variant",    f"{n_per_variant:,}", "users needed"),
        (c2, "Total sample",   f"{n_total:,}",       "across both variants"),
        (c3, "Days to run",    f"{days_needed}",      f"at {daily_v:,} visitors/day"),
        (c4, "Detectable rate",f"{baseline*(1+mde_pct/100):.2%}", f"from baseline {baseline:.2%}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-label">{label}</div>
              <div class="card-value" style="color:{COL_B}">{val}</div>
              <div class="card-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Days needed vs MDE (tradeoff curve)")

    mdes   = np.linspace(5, 100, 100)
    days_c = [int(np.ceil(required_sample_size(baseline, m/100, alpha, power) * 2 / daily_v))
              for m in mdes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mdes, y=days_c, mode="lines",
        line=dict(color=COL_B, width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
    ))
    fig.add_vline(x=mde_pct, line_dash="dash", line_color=COL_A,
                  annotation_text=f"Your MDE: {mde_pct}%",
                  annotation_position="top right")

    fig.update_layout(
        xaxis=dict(title="Min Detectable Effect (% relative lift)", gridcolor=GRID),
        yaxis=dict(title="Days Required", gridcolor=GRID),
        plot_bgcolor=CARD, paper_bgcolor=BG, font=FONT,
        margin=dict(l=10,r=10,t=20,b=10), height=340,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Smaller effects require exponentially more data. If you want to detect a 5% relative lift "
        "but can only run for 2 weeks, you're underpowered — you might miss real effects."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#352e47;font-size:12px;font-family:JetBrains Mono'>"
    "Beta-Binomial conjugacy · Monte Carlo posterior sampling · Expected loss decision criterion"
    "</div>", unsafe_allow_html=True,
)
