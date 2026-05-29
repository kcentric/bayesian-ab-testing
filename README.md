# Bayesian A/B Testing: A Principled Framework for Conversion Experiments

A statistical write-up and interactive tool for running Bayesian inference on
A/B test data — and an explanation of why the standard frequentist approach
answers the wrong question.

---

## The Problem with p-values

When a product team runs an A/B test, they want to know:

> *"What is the probability that variant B has a higher true conversion rate than A?"*

The frequentist p-value cannot answer this. A p-value is:

> P(data this extreme or more extreme | H₀ is true)

That is: *assuming no difference exists*, how surprising is our data? This is almost
never what anyone actually wants to know, and it's routinely misinterpreted as the
probability that the null hypothesis is false.

**Concrete example:** A p-value of 0.03 does **not** mean there's a 97% chance B is
better. It means: if A and B were identical, we'd observe data this extreme only 3%
of the time. These are very different statements.

---

## The Bayesian Model

We model the unknown true conversion rate θ as a Beta-distributed random variable.

**Why Beta?** The Beta distribution is defined on [0,1] — the natural support for
a probability — and it is conjugate to the Binomial likelihood, meaning the posterior
is analytically tractable.

```
Prior:      θ ~ Beta(α₀, β₀)
Likelihood: k | θ, n ~ Binomial(n, θ)       [k conversions from n visitors]
Posterior:  θ | k, n ~ Beta(α₀ + k, β₀ + n - k)
```

This is **Beta-Binomial conjugacy** — the posterior is the same distributional family
as the prior, updated simply by adding observed counts. No MCMC, no numerical integration.

We use an **uninformative prior** Beta(1,1) — the uniform distribution over [0,1],
expressing no prior belief about the conversion rate. As data accumulates, the posterior
tightens around the true value.

---

## What We Can Compute

### P(B > A)
The probability that variant B's true conversion rate exceeds A's. Computed via:

1. **Analytical formula** (exact for integer parameters, O(β_B) time):
   Uses the closed-form Beta inequality formula (Cook 2005).

2. **Monte Carlo** (general fallback): Draw 100k samples from each posterior,
   compute fraction where B > A. Standard error ~0.002 at 100k samples.

### Credible Intervals
The 95% credible interval is the range we're 95% confident contains the true rate.
Unlike a frequentist confidence interval, this statement is literally true — it is
a direct probability statement about θ given our data.

### Expected Loss
The expected loss of choosing variant B (when A might actually be better):

```
E[Loss | choose B] = E[max(θ_A - θ_B, 0)]
```

Computed via Monte Carlo. We stop the experiment when the minimum expected loss
falls below a business-defined threshold (e.g. 0.5% of conversion rate).

This is more principled than a fixed significance threshold — it incorporates
both the direction and *magnitude* of the potential error.

### Relative Uplift Distribution
The full posterior distribution of (θ_B - θ_A) / θ_A — the relative lift.
Allows statements like: *"there's a 95% probability that B improves conversion
by between 8% and 34%."* This is what a product manager can act on.

---

## Bayesian vs Frequentist: Direct Comparison

| Question | Frequentist | Bayesian |
|---|---|---|
| P(B > A)? | ❌ Cannot compute | ✅ Direct answer |
| How big is the effect? | CI on difference | Full posterior of uplift |
| When to stop? | Fixed sample size required | Sequential: stop when P(B>A) > threshold |
| Intuitive to stakeholders? | Rarely | Generally yes |
| Regulatory acceptance? | Standard | Sometimes required to use frequentist |

**When frequentist wins:** Pre-registered trials, regulatory environments, fixed budgets
requiring strict type-I error control.

**When Bayesian wins:** Sequential testing with early stopping, communicating results
to non-technical stakeholders, incorporating prior knowledge from previous experiments,
making decisions under uncertainty with explicit loss functions.

---

## Dashboard (5 pages)

| Page | Content |
|---|---|
| **The Method** | Visual explanation of Beta posteriors updating with data |
| **Run an Experiment** | Enter observed data → full Bayesian inference output |
| **Live Simulation** | Day-by-day simulation showing P(B>A) and credible intervals evolving |
| **Bayes vs Frequentist** | Same data, both frameworks, side-by-side comparison |
| **Sample Size Calculator** | Power/MDE tradeoff curve — when are you adequately powered? |

---

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
bayesian_ab/
├── app.py              # 5-page Streamlit dashboard
├── src/
│   └── bayesian_ab.py  # Full statistical engine
└── README.md
```

---

## Skills Demonstrated

- **Bayesian inference** — Beta-Binomial conjugacy, posterior updating, credible intervals
- **Probability theory** — Beta distribution, analytical inequality formula (Cook 2005)
- **Monte Carlo methods** — posterior sampling, expected loss estimation
- **Decision theory** — expected loss as a stopping criterion vs fixed significance thresholds
- **Statistical communication** — explaining what p-values do and don't mean
- **Experiment design** — power analysis, MDE, sample size calculation
- **Python** — NumPy, SciPy, Plotly, Streamlit, dataclasses

---

## References

- Cook, J. (2005). *Exact calculation of beta inequalities.* UT MD Anderson Technical Report.
- Deng, A. et al. (2016). *Continuous monitoring of A/B tests without pain.* IEEE DSAA.
- VanderPlas, J. (2014). *Frequentism and Bayesianism: A Python-driven Primer.*
