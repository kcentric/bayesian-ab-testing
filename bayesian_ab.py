"""
bayesian_ab.py
==============
Bayesian A/B Testing for Conversion Rate Experiments

WHY BAYESIAN INSTEAD OF FREQUENTIST?
─────────────────────────────────────
The standard frequentist approach gives you a p-value — the probability of seeing
data this extreme *if the null hypothesis were true*. This is:
  a) Not what anyone actually wants to know
  b) Routinely misinterpreted as "probability that B is better than A"
  c) Gives no guidance on HOW MUCH better B is, or whether the difference is
     worth acting on

Bayesian inference gives you what you actually want:
  - P(B > A) — the probability that variant B has a higher true conversion rate
  - The full posterior distribution over possible conversion rates
  - Expected loss — the cost of making the wrong decision
  - Credible intervals — "there's a 95% chance the true rate is in this range"
    (unlike confidence intervals, this is actually what the name implies)

THE MODEL
──────────
We model conversion rates as Beta distributions:
  - Prior: Beta(α₀, β₀) — our belief before seeing any data
    (Beta(1,1) = uniform, i.e. we have no prior information)
  - Likelihood: each visitor either converts (Bernoulli trial) or doesn't
  - Posterior: Beta(α₀ + conversions, β₀ + non-conversions)
    — the Beta-Binomial conjugacy means the posterior is analytically tractable

The Beta(α, β) distribution has:
  - Mean = α / (α + β)         ← our best estimate of the true rate
  - Variance = αβ / (α+β)²(α+β+1) ← our uncertainty about that estimate

More data → tighter posterior → more confident conclusion.

DECISION CRITERIA
──────────────────
We stop (make a decision) when:
  1. P(B > A) > threshold (e.g. 95%) — confident B is better
  2. P(A > B) > threshold            — confident A is better
  3. Expected Loss < ε               — the potential downside of being wrong
     is smaller than a business-defined threshold
"""

import numpy as np
from scipy import stats
from scipy.special import betaln
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 1.  BETA DISTRIBUTION HELPERS
# ─────────────────────────────────────────────

@dataclass
class BetaPosterior:
    """
    Represents the posterior belief about a conversion rate after seeing data.

    Parameters
    ----------
    alpha : float  — prior α + observed conversions
    beta  : float  — prior β + observed non-conversions
    name  : str    — label for plotting
    """
    alpha: float
    beta:  float
    name:  str = ""

    @property
    def mean(self) -> float:
        """Expected conversion rate under this posterior."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n ** 2 * (n + 1))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        HDI (Highest Density Interval) at the given level.
        This is the Bayesian analogue of a confidence interval — but unlike
        a CI, we CAN say "there's a 95% probability the true rate lies here."
        """
        lo = (1 - level) / 2
        hi = 1 - lo
        return (
            float(stats.beta.ppf(lo, self.alpha, self.beta)),
            float(stats.beta.ppf(hi, self.alpha, self.beta)),
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function over [0,1]."""
        return stats.beta.pdf(x, self.alpha, self.beta)

    def sample(self, n: int = 100_000, seed: int = None) -> np.ndarray:
        """Draw samples from the posterior — used for Monte Carlo estimation."""
        rng = np.random.default_rng(seed)
        return rng.beta(self.alpha, self.beta, n)


# ─────────────────────────────────────────────
# 2.  CORE INFERENCE FUNCTIONS
# ─────────────────────────────────────────────

def update_posterior(prior_alpha: float, prior_beta: float,
                     conversions: int, visitors: int) -> BetaPosterior:
    """
    Bayesian update: incorporate observed data into prior belief.

    By Beta-Binomial conjugacy:
      posterior = Beta(α_prior + conversions, β_prior + non-conversions)

    This is the analytical solution — no MCMC or numerical integration needed.
    """
    non_conversions = visitors - conversions
    return BetaPosterior(
        alpha=prior_alpha + conversions,
        beta= prior_beta  + non_conversions,
    )


def prob_b_beats_a_analytical(a: BetaPosterior, b: BetaPosterior) -> float:
    """
    Compute P(B > A) analytically using the closed-form formula for
    the probability that one Beta-distributed variable exceeds another.

    The formula sums over the probability mass of B being in each
    interval where it exceeds A. For Beta distributions:

    P(B > A) = Σ_{i=0}^{β_B-1}  exp(
                  log B(α_A + α_B + i, β_A + β_B - 1 - i)
                  - log(β_B - 1 - i)
                  - log B(1 + i, β_B - 1 - i)
                  - log B(α_A, β_A)
               )

    This is exact for integer α, β. For non-integers we fall back to Monte Carlo.
    (Reference: Cook 2005, "Exact calculation of beta inequalities")
    """
    # Fall back to Monte Carlo for non-integer or large parameters
    if (a.alpha % 1 != 0 or a.beta % 1 != 0 or
        b.alpha % 1 != 0 or b.beta % 1 != 0 or
        a.alpha + a.beta > 3000 or b.alpha + b.beta > 3000):
        return prob_b_beats_a_mc(a, b)

    alpha_a, beta_a = int(a.alpha), int(a.beta)
    alpha_b, beta_b = int(b.alpha), int(b.beta)

    total = 0.0
    for i in range(beta_b):
        total += np.exp(
            betaln(alpha_a + alpha_b + i, beta_a + beta_b - 1 - i)
            - np.log(beta_b - 1 - i + 1e-10)
            - betaln(1 + i, beta_b - 1 - i)
            - betaln(alpha_a, beta_a)
        )
    return float(np.clip(total, 0, 1))


def prob_b_beats_a_mc(a: BetaPosterior, b: BetaPosterior,
                      n: int = 100_000, seed: int = 42) -> float:
    """
    Monte Carlo estimate of P(B > A).

    Draw n samples from each posterior and compute the fraction where B > A.
    With 100k samples, the standard error is ~0.0016 — more than precise enough.
    """
    samples_a = a.sample(n, seed=seed)
    samples_b = b.sample(n, seed=seed + 1)
    return float((samples_b > samples_a).mean())


def expected_loss(a: BetaPosterior, b: BetaPosterior,
                  n: int = 100_000, seed: int = 42) -> dict:
    """
    Expected loss of choosing each variant.

    If we choose B but A is actually better, our loss is max(θ_A - θ_B, 0).
    If we choose A but B is actually better, our loss is max(θ_B - θ_A, 0).

    We want to stop the experiment when the expected loss of making the
    wrong decision falls below a business-defined threshold (e.g. 0.001 = 0.1%).

    This is more principled than a fixed significance threshold because it
    incorporates the magnitude of the difference, not just the direction.
    """
    samples_a = a.sample(n, seed=seed)
    samples_b = b.sample(n, seed=seed + 1)

    # Loss if we pick B (and A is actually better)
    loss_b = float(np.maximum(samples_a - samples_b, 0).mean())
    # Loss if we pick A (and B is actually better)
    loss_a = float(np.maximum(samples_b - samples_a, 0).mean())

    return {"loss_if_choose_a": loss_a, "loss_if_choose_b": loss_b}


def relative_uplift(a: BetaPosterior, b: BetaPosterior,
                    n: int = 100_000, seed: int = 42) -> dict:
    """
    Posterior distribution of the relative uplift: (θ_B - θ_A) / θ_A.

    Returns percentiles for the credible interval on the uplift.
    This tells you not just WHETHER B is better but HOW MUCH better,
    in the units a product manager understands: "lift over control."
    """
    samples_a = a.sample(n, seed=seed)
    samples_b = b.sample(n, seed=seed + 1)
    relative   = (samples_b - samples_a) / (samples_a + 1e-10)

    return {
        "mean":   float(relative.mean()),
        "p2_5":   float(np.percentile(relative, 2.5)),
        "p25":    float(np.percentile(relative, 25)),
        "p50":    float(np.percentile(relative, 50)),
        "p75":    float(np.percentile(relative, 75)),
        "p97_5":  float(np.percentile(relative, 97.5)),
        "samples": relative.tolist(),
    }


# ─────────────────────────────────────────────
# 3.  EXPERIMENT SIMULATOR
# ─────────────────────────────────────────────

def simulate_experiment(true_rate_a: float, true_rate_b: float,
                        daily_visitors: int = 500,
                        n_days: int = 30,
                        prior_alpha: float = 1.0,
                        prior_beta:  float = 1.0,
                        decision_threshold: float = 0.95,
                        loss_threshold: float = 0.005,
                        seed: int = 42) -> dict:
    """
    Simulate a live A/B test day-by-day, updating posteriors as data arrives.

    Each day:
      1. Simulate visitors split equally between A and B
      2. Each visitor converts with their variant's true rate (Bernoulli)
      3. Update posteriors for both variants
      4. Compute P(B > A) and expected loss
      5. Check if we've reached a decision

    Returns the full history for visualisation.
    """

    rng = np.random.default_rng(seed)

    # Cumulative counters
    visitors_a = visitors_b = 0
    conv_a     = conv_b     = 0

    history = []
    decision = None

    for day in range(1, n_days + 1):
        # Simulate today's visitors (split 50/50)
        v_a = daily_visitors // 2
        v_b = daily_visitors - v_a

        # Each visitor converts independently with their variant's true rate
        conv_today_a = int(rng.binomial(v_a, true_rate_a))
        conv_today_b = int(rng.binomial(v_b, true_rate_b))

        visitors_a += v_a;  conv_a += conv_today_a
        visitors_b += v_b;  conv_b += conv_today_b

        # Update posteriors
        post_a = update_posterior(prior_alpha, prior_beta, conv_a, visitors_a)
        post_b = update_posterior(prior_alpha, prior_beta, conv_b, visitors_b)
        post_a.name = "Control (A)"
        post_b.name = "Variant (B)"

        # Key metrics
        p_b_wins = prob_b_beats_a_mc(post_a, post_b, n=50_000, seed=day)
        loss     = expected_loss(post_a, post_b, n=50_000, seed=day)
        ci_a     = post_a.credible_interval()
        ci_b     = post_b.credible_interval()

        record = {
            "day":           day,
            "visitors_a":    visitors_a,
            "visitors_b":    visitors_b,
            "conv_a":        conv_a,
            "conv_b":        conv_b,
            "rate_a":        conv_a / visitors_a,
            "rate_b":        conv_b / visitors_b,
            "post_a_mean":   post_a.mean,
            "post_b_mean":   post_b.mean,
            "post_a_alpha":  post_a.alpha,
            "post_a_beta":   post_a.beta,
            "post_b_alpha":  post_b.alpha,
            "post_b_beta":   post_b.beta,
            "p_b_wins":      p_b_wins,
            "loss_choose_a": loss["loss_if_choose_a"],
            "loss_choose_b": loss["loss_if_choose_b"],
            "ci_a_lo":       ci_a[0], "ci_a_hi": ci_a[1],
            "ci_b_lo":       ci_b[0], "ci_b_hi": ci_b[1],
            "decision":      None,
        }

        # Decision rule
        if decision is None:
            if p_b_wins >= decision_threshold:
                decision = "B wins"
                record["decision"] = "B wins"
            elif (1 - p_b_wins) >= decision_threshold:
                decision = "A wins"
                record["decision"] = "A wins"
            elif (loss["loss_if_choose_b"] < loss_threshold and
                  loss["loss_if_choose_a"] < loss_threshold):
                decision = "No meaningful difference"
                record["decision"] = "No meaningful difference"

        history.append(record)

    # Final posteriors for distribution plot
    final_post_a = update_posterior(prior_alpha, prior_beta, conv_a, visitors_a)
    final_post_b = update_posterior(prior_alpha, prior_beta, conv_b, visitors_b)
    final_post_a.name = "Control (A)"
    final_post_b.name = "Variant (B)"

    uplift = relative_uplift(final_post_a, final_post_b)

    return {
        "history":     history,
        "final_post_a": {"alpha": final_post_a.alpha, "beta": final_post_a.beta,
                         "mean": final_post_a.mean, "name": "Control (A)"},
        "final_post_b": {"alpha": final_post_b.alpha, "beta": final_post_b.beta,
                         "mean": final_post_b.mean, "name": "Variant (B)"},
        "decision":     decision or "Inconclusive",
        "uplift":       uplift,
        "true_rate_a":  true_rate_a,
        "true_rate_b":  true_rate_b,
    }


# ─────────────────────────────────────────────
# 4.  FREQUENTIST COMPARISON
# ─────────────────────────────────────────────

def frequentist_test(conv_a: int, visitors_a: int,
                     conv_b: int, visitors_b: int) -> dict:
    """
    Run the equivalent frequentist two-proportion z-test.
    We include this to show the contrast — and why it's less informative.

    The z-test gives:
      - p-value: P(data this extreme | H0: rates are equal)
      - It does NOT tell you P(B > A)
      - It does NOT tell you how big the difference is
      - It depends on sample size in a way that can be gamed

    We show both tests side by side so the viewer can see what each actually answers.
    """
    rate_a = conv_a / visitors_a
    rate_b = conv_b / visitors_b

    # Pooled proportion under H0
    pool   = (conv_a + conv_b) / (visitors_a + visitors_b)
    se     = np.sqrt(pool * (1 - pool) * (1/visitors_a + 1/visitors_b))
    z_stat = (rate_b - rate_a) / (se + 1e-10)
    p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    # 95% CI on the difference (frequentist)
    se_diff = np.sqrt(rate_a*(1-rate_a)/visitors_a + rate_b*(1-rate_b)/visitors_b)
    diff_lo = (rate_b - rate_a) - 1.96 * se_diff
    diff_hi = (rate_b - rate_a) + 1.96 * se_diff

    return {
        "rate_a":    rate_a,
        "rate_b":    rate_b,
        "z_stat":    z_stat,
        "p_value":   p_val,
        "significant": p_val < 0.05,
        "diff_lo":   diff_lo,
        "diff_hi":   diff_hi,
        "absolute_diff": rate_b - rate_a,
    }


# ─────────────────────────────────────────────
# 5.  SAMPLE SIZE / POWER CALCULATOR
# ─────────────────────────────────────────────

def required_sample_size(baseline_rate: float, min_detectable_effect: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Frequentist sample size calculation (per variant).

    How many users do we need per variant to detect an effect of size MDE
    with the given power at significance level alpha?

    This is the standard formula used to plan experiments before launch.
    It shows you understand both frameworks — Bayesian for inference,
    frequentist for planning.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1)) +
          z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2) / ((p2 - p1) ** 2)

    return int(np.ceil(n))


# ─────────────────────────────────────────────
# 6.  SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Bayesian A/B Test ===")
    print("Scenario: true rates A=5%, B=6.5% (30% relative lift)\n")

    result = simulate_experiment(
        true_rate_a=0.05, true_rate_b=0.065,
        daily_visitors=500, n_days=30,
        decision_threshold=0.95,
    )

    h = result["history"]
    final = h[-1]
    print(f"After {len(h)} days:")
    print(f"  A: {final['conv_a']:,} conversions / {final['visitors_a']:,} visitors "
          f"= {final['rate_a']:.2%}")
    print(f"  B: {final['conv_b']:,} conversions / {final['visitors_b']:,} visitors "
          f"= {final['rate_b']:.2%}")
    print(f"  P(B > A): {final['p_b_wins']:.1%}")
    print(f"  Decision: {result['decision']}")
    print(f"  Relative uplift (median): {result['uplift']['p50']:.1%}")
    print(f"  95% credible interval on uplift: "
          f"[{result['uplift']['p2_5']:.1%}, {result['uplift']['p97_5']:.1%}]")

    freq = frequentist_test(final['conv_a'], final['visitors_a'],
                            final['conv_b'], final['visitors_b'])
    print(f"\n=== Frequentist comparison ===")
    print(f"  p-value: {freq['p_value']:.4f} ({'significant' if freq['significant'] else 'not significant'})")
    print(f"  95% CI on difference: [{freq['diff_lo']:.4f}, {freq['diff_hi']:.4f}]")

    n = required_sample_size(0.05, 0.30)
    print(f"\n=== Sample size needed ===")
    print(f"  To detect 30% lift on 5% baseline with 80% power: {n:,} per variant")
