# ==============================================================================
#                             IMPORTS (OPTIMIZED)
# ==============================================================================
import streamlit as st
import numpy as np
import math
import pandas as pd
from scipy.stats import norm
from typing import Optional, Dict, List, Any
from statsmodels.stats.power import TTestIndPower, TTestPower, FTestAnovaPower

# ==============================================================================
#                             CONSTANTS & CONFIG
# ==============================================================================
# Asymptotic Relative Efficiency (ARE) factors
ARE_FACTORS = {"wilcoxon": 0.955, "mann_whitney": 0.955, "kruskal_wallis": 0.955}
FISHER_ADJUSTMENTS = {"power": 0.95, "n": 1.05}

# Practical guidance constants
RECRUITMENT_FEASIBILITY_THRESHOLDS = {
    "easy": 100,
    "moderate": 500,
    "challenging": 1000,
    "very_difficult": 5000
}

# Common statistical pitfalls and warnings
COMMON_PITFALLS = {
    "small_sample": "Small sample sizes may violate test assumptions. Consider pilot studies.",
    "large_effect": "Very large effect sizes may be unrealistic. Verify with pilot data or literature.",
    "small_effect": "Small effect sizes require very large samples. Consider clinical significance.",
    "low_power": "Power below 0.80 may miss clinically important effects.",
    "high_alpha": "Alpha > 0.05 increases Type I error risk. Justify carefully.",
    "equal_allocation": "Unequal allocation may be more efficient with different costs or variances.",
}

# Intra-Cluster Correlation (ICC) typical ranges by field
ICC_RANGES = {
    "primary_care": (0.01, 0.05),
    "schools": (0.05, 0.20),
    "hospitals": (0.01, 0.10),
    "communities": (0.001, 0.05),
    "worksites": (0.01, 0.10),
}

# Bayesian prior distributions for common effect sizes
BAYESIAN_PRIORS = {
    "skeptical": {"mean": 0.1, "sd": 0.15},  # Skeptical prior (small effects)
    "optimistic": {"mean": 0.5, "sd": 0.25},  # Optimistic prior (medium effects)
    "neutral": {"mean": 0.3, "sd": 0.30},     # Neutral/Vague prior
    "enthusiastic": {"mean": 0.8, "sd": 0.30}, # Enthusiastic prior (large effects)
}

# Centralized citations
CITATIONS = {
    "cohen_1988": (
        "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum Associates.",
        "https://www.utstat.toronto.edu/~brunner/oldclass/378f16/readings/CohenPower.pdf"),
    "statsmodels_ttestind": ("Statsmodels Documentation: TTestIndPower",
                             "https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.html"),
    "statsmodels_ttest": ("Statsmodels Documentation: TTestPower (One-Sample/Paired)",
                          "https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html"),
    "statsmodels_prop": ("Statsmodels Documentation: power_proportions_2indep",
                         "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html"),
    "statsmodels_anova": ("Statsmodels Documentation: FTestAnovaPower",
                          "https://www.statsmodels.org/stable/generated/statsmodels.stats.power.FTestAnovaPower.html"),
    "ich_e9": ("ICH E9: Statistical Principles for Clinical Trials (1998).",
               "https://database.ich.org/sites/default/files/E9_Guideline.pdf"),
    "basic_biostats": ("General concepts from introductory Biostatistics.",
                       "https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one"),
    "cohen_h_info": ("Wikipedia: Cohen's h", "https://en.wikipedia.org/wiki/Cohen%27s_h"),
    "cohen_f_info": ("Wikipedia: Effect size#Cohen's_f", "https://en.wikipedia.org/wiki/Effect_size#Cohen's_f"),
    "are_info": ("Wikipedia: Asymptotic Relative Efficiency (ARE)",
                 "https://en.wikipedia.org/wiki/Asymptotic_relative_efficiency"),
    "wilcoxon_signed_rank": ("Statistics How To: Wilcoxon Signed-Rank Test",
                             "https://www.statisticshowto.com/wilcoxon-signed-rank-test/"),
    "mann_whitney": ("Statistics How To: Mann-Whitney U Test",
                     "https://www.statisticshowto.com/mann-whitney-u-test/"),
    "kruskal_wallis": ("Statistics How To: Kruskal-Wallis H Test",
                       "https://www.statisticshowto.com/kruskal-wallis/"),
    "fishers_exact": ("Statistics How To: Fisher's Exact Test",
                      "https://www.statisticshowto.com/fishers-exact-test/"),
    "effect_size_estimation": ("Statistics Solutions: Effect Size",
                               "https://www.statisticssolutions.com/effect-size/"),
    "normal_approx_prop": ("PennState STAT 500: Normal Approximation for Binomial",
                           "https://online.stat.psu.edu/stat500/lesson/5/5.3/5.3.1"),
    "schoenfeld": ("Schoenfeld, D. (1981). The asymptotic properties of nonparametric tests for comparing survival distributions. Biometrika, 68(1), 316-319.",
                   "https://academic.oup.com/biomet/article-abstract/68/1/316/263254"),
    "logrank_test": ("Wikipedia: Log-rank test",
                     "https://en.wikipedia.org/wiki/Logrank_test"),
    "hazard_ratio": ("Hazard Ratio - Statistics How To",
                     "https://www.statisticshowto.com/hazard-ratio/"),
    "donner_klar": ("Donner, A., & Klar, N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. Arnold.",
                    "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118763452"),
    "icc_info": ("Intraclass Correlation - Design Effect",
                 "https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_power/bs704_power_print.html"),
    "repeated_measures": ("Repeated Measures ANOVA - Statistical Power",
                          "https://www.statisticshowto.com/repeated-measures-anova/"),
    "bayesian_sample_size": ("Spiegelhalter, D. J., et al. (2004). Bayesian approaches to randomized trials. JRSS-A, 167(3), 357-416.",
                             "https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-985X.2004.02044.x"),
    "assurance": ("O'Hagan, A., et al. (2005). Assurance in clinical trial design. Pharmaceutical Statistics, 4(3), 187-201.",
                  "https://onlinelibrary.wiley.com/doi/abs/10.1002/pst.175"),
}


# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================
def calculate_effect_size(effect_type: str, **kwargs) -> Optional[float]:
    """Unified effect size calculator with improved error handling."""
    try:
        if effect_type == "cohen_d_two":
            mean1, mean2, pooled_sd = kwargs.get('mean1'), kwargs.get('mean2'), kwargs.get('pooled_sd')
            if all(v is not None for v in [mean1, mean2, pooled_sd]) and pooled_sd > 0:
                return abs(mean1 - mean2) / pooled_sd
        elif effect_type == "cohen_d_one":
            sample_mean, hyp_mean, sd = kwargs.get('sample_mean'), kwargs.get('hypothesized_mean'), kwargs.get('sd')
            if all(v is not None for v in [sample_mean, hyp_mean, sd]) and sd > 0:
                return abs(sample_mean - hyp_mean) / sd
        elif effect_type == "cohen_d_paired":
            mean_diff, sd_diff = kwargs.get('mean_diff'), kwargs.get('sd_diff')
            if all(v is not None for v in [mean_diff, sd_diff]) and sd_diff > 0:
                return abs(mean_diff) / sd_diff
        elif effect_type == "cohen_h":
            p1, p2 = kwargs.get('p1'), kwargs.get('p2')
            if all(v is not None for v in [p1, p2]) and 0 < p1 < 1 and 0 < p2 < 1 and p1 != p2:
                return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    except Exception as e:
        st.error(f"Error calculating effect size: {e}")
    return None


def display_explanation(header: str, content: str, citation_key: Optional[str] = None,
                        help_text: Optional[str] = None, expanded: bool = False) -> None:
    """Display expandable explanations with citations."""
    with st.expander(header, expanded=expanded):
        if help_text:
            st.caption(f":information_source: {help_text}")
        st.markdown(content, unsafe_allow_html=True)
        if citation_key and citation_key in CITATIONS:
            text, url = CITATIONS[citation_key]
            st.caption(f"Source: [{text}]({url})")


def display_results_table(data: Dict[str, List[Any]]) -> None:
    """Display results summary table."""
    try:
        df = pd.DataFrame(data)
        if "Value" in df.columns:
            df["Value"] = df["Value"].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        st.dataframe(df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying table: {e}")


def check_expected_counts(test_type: str, n1: float, n_ratio: float, raw_inputs: Dict, show_warning: bool = False):
    """Check expected counts for proportion tests with enhanced guidance."""
    if test_type == "two_prop":
        p1, p2 = raw_inputs.get("prop1"), raw_inputs.get("prop2")
        if n1 is not None and p1 is not None and p2 is not None:
            n2 = math.ceil(n1 * n_ratio)
            counts = [n1 * p1, n1 * (1 - p1), n2 * p2, n2 * (1 - p2)]
            min_count = min(counts)
            if min_count < 5:
                msg = f"⚠️ Expected cell count < 5 ({min_count:.1f}). The normal approximation may be inaccurate. Consider Fisher's Exact Test for small samples."
                (st.warning if show_warning else st.sidebar.warning)(msg)
            elif min_count < 10:
                msg = f"ℹ️ Expected cell count between 5-10 ({min_count:.1f}). Results are approximate. Consider using exact methods if possible."
                (st.info if show_warning else st.sidebar.info)(msg)
    elif test_type == "one_prop":
        null_p = raw_inputs.get("null_prop")
        if n1 is not None and null_p is not None:
            counts = [n1 * null_p, n1 * (1 - null_p)]
            min_count = min(counts)
            if min_count < 5:
                msg = f"⚠️ Expected count < 5 ({min_count:.1f}). Z-test approximation may be unreliable. Consider exact binomial methods."
                (st.warning if show_warning else st.sidebar.warning)(msg)
            elif min_count < 10:
                msg = f"ℹ️ Expected count between 5-10 ({min_count:.1f}). Normal approximation is approximate."
                (st.info if show_warning else st.sidebar.info)(msg)


def validate_effect_size(effect_size: float, effect_type: str, test_name: str) -> List[str]:
    """Validate effect size and return warnings/suggestions."""
    warnings = []

    if effect_size is None or effect_size <= 0:
        warnings.append("❌ Effect size must be positive.")
        return warnings

    # Check for unrealistically large effects
    if effect_type in ["cohen_d_two", "cohen_d_one", "cohen_d_paired"]:
        if effect_size > 2.0:
            warnings.append(f"⚠️ Very large effect size ({effect_size:.2f}). Cohen's d > 2.0 is rare in most fields. Verify this is realistic for your context.")
        elif effect_size < 0.1:
            warnings.append(f"ℹ️ Very small effect size ({effect_size:.2f}). This will require a very large sample. Consider whether such a small effect is clinically/practically meaningful.")

    elif effect_type == "cohen_f":
        if effect_size > 0.8:
            warnings.append(f"⚠️ Very large effect size ({effect_size:.2f}). Cohen's f > 0.8 is uncommon. Verify with pilot data or literature.")
        elif effect_size < 0.05:
            warnings.append(f"ℹ️ Very small effect size ({effect_size:.2f}). Large sample sizes will be required.")

    elif effect_type == "cohen_h":
        if effect_size > 1.5:
            warnings.append(f"⚠️ Very large effect size ({effect_size:.2f}). This represents a very large difference in proportions. Verify this is realistic.")

    elif effect_type == "hazard_ratio":
        if effect_size < 0.3 or effect_size > 3.0:
            warnings.append(f"⚠️ Extreme hazard ratio ({effect_size:.2f}). Verify this is clinically plausible for your context.")

    return warnings


def assess_recruitment_feasibility(total_n: int, test_name: str) -> str:
    """Provide recruitment feasibility assessment."""
    if total_n <= RECRUITMENT_FEASIBILITY_THRESHOLDS["easy"]:
        return f"✅ **Recruitment Feasibility: Easy** (N={total_n}). This sample size is generally achievable in most settings."
    elif total_n <= RECRUITMENT_FEASIBILITY_THRESHOLDS["moderate"]:
        return f"⚠️ **Recruitment Feasibility: Moderate** (N={total_n}). May require multi-site recruitment or extended timeline. Consider 12-24 months for recruitment."
    elif total_n <= RECRUITMENT_FEASIBILITY_THRESHOLDS["challenging"]:
        return f"⚠️ **Recruitment Feasibility: Challenging** (N={total_n}). Likely requires multi-center collaboration. Consider 24-36 months for recruitment. May need to adjust design or effect size expectations."
    elif total_n <= RECRUITMENT_FEASIBILITY_THRESHOLDS["very_difficult"]:
        return f"🚨 **Recruitment Feasibility: Very Difficult** (N={total_n}). May be infeasible for single studies. Consider: (1) consortium approach, (2) registry-based studies, (3) adaptive designs, or (4) revising assumptions."
    else:
        return f"🚨 **Recruitment Feasibility: Extremely Difficult** (N={total_n}). This sample size is rarely achievable. Strongly consider alternative study designs, Bayesian approaches, or meta-analytic frameworks."


def estimate_study_timeline(total_n: int, monthly_recruitment_rate: int = None) -> str:
    """Estimate study timeline based on sample size."""
    if monthly_recruitment_rate is None:
        # Provide rough guidelines
        if total_n <= 50:
            return "Estimated timeline: 6-12 months (assuming typical single-site recruitment)"
        elif total_n <= 200:
            return "Estimated timeline: 12-24 months (assuming single or dual-site recruitment)"
        elif total_n <= 500:
            return "Estimated timeline: 24-36 months (likely multi-site recruitment needed)"
        else:
            return "Estimated timeline: 36+ months (multi-center collaboration required)"
    else:
        months = math.ceil(total_n / monthly_recruitment_rate)
        years = months / 12
        if years < 1:
            return f"Estimated recruitment duration: {months} months (at {monthly_recruitment_rate} participants/month)"
        else:
            return f"Estimated recruitment duration: {years:.1f} years ({months} months) at {monthly_recruitment_rate} participants/month"


def generate_sample_size_justification(config: Dict, inputs: Dict, result: float) -> str:
    """Generate sample size justification text for protocols."""
    goal = inputs["goal"]
    if goal != "Sample Size":
        return ""

    n1 = int(np.ceil(result))
    n_ratio = inputs.get("n_ratio", 1.0)
    n2 = int(np.ceil(n1 * n_ratio)) if config.get("n_ratio") else None
    k = inputs.get("k_groups", 1 if not n2 else 2)
    total = n1 * k if not n2 else n1 + n2

    test_name = st.session_state.get("selected_test", "Statistical Test")
    alpha = inputs["alpha"]
    power = inputs.get("power", 0.80)
    effect = inputs.get("effect_size")
    alt = inputs["alternative"]
    objective = inputs.get("objective", "Superiority")
    dropout = inputs.get("dropout", 0)

    # Build justification text
    text = f"""
### Sample Size Justification

The sample size calculation for this {objective.lower()} study using a **{test_name}** is based on the following assumptions:

**Statistical Parameters:**
- Significance level (α): {alpha} ({alt} test)
- Desired statistical power (1-β): {power:.0%}
- Expected effect size: {effect:.3f} {_get_effect_size_interpretation(config.get('effect'), effect)}

**Sample Size:**
"""

    if n2:
        text += f"- Group 1: {n1} participants\n"
        text += f"- Group 2: {n2} participants\n"
        text += f"- **Total: {total} participants**\n"
    elif k > 2:
        text += f"- Per group: {n1} participants\n"
        text += f"- Number of groups: {k}\n"
        text += f"- **Total: {total} participants**\n"
    else:
        text += f"- **Total: {n1} participants**\n"

    if dropout > 0:
        n1_adj = int(np.ceil(n1 / (1 - dropout / 100)))
        total_adj = n1_adj * k if not n2 else n1_adj + int(np.ceil(n1_adj * n_ratio))
        text += f"\n**Adjusted for {dropout}% expected dropout:**\n"
        text += f"- **Total recruitment target: {total_adj} participants**\n"

    text += f"""
**Justification:**
This sample size provides {power:.0%} power to detect an effect size of {effect:.3f}, which {_interpret_clinical_significance(effect, config.get('effect'))}, using a {alt} test at the {alpha} significance level.
"""

    return text


def _get_effect_size_interpretation(effect_type: str, value: float) -> str:
    """Get interpretation of effect size type."""
    interpretations = {
        "cohen_d_two": f"(Cohen's d: difference between groups)",
        "cohen_d_one": f"(Cohen's d: difference from null value)",
        "cohen_d_paired": f"(Cohen's d: paired difference)",
        "cohen_h": f"(Cohen's h: difference in proportions)",
        "cohen_f": f"(Cohen's f: ANOVA effect)",
        "hazard_ratio": f"(Hazard Ratio)",
        "wilcoxon_special": f"(standardized difference)"
    }
    return interpretations.get(effect_type, "")


def _interpret_clinical_significance(effect_size: float, effect_type: str) -> str:
    """Interpret clinical/practical significance of effect size."""
    if effect_type in ["cohen_d_two", "cohen_d_one", "cohen_d_paired", "wilcoxon_special"]:
        if effect_size < 0.2:
            return "represents a very small effect that may have limited clinical significance"
        elif effect_size < 0.5:
            return "represents a small to medium effect of potential clinical relevance"
        elif effect_size < 0.8:
            return "represents a medium to large effect of likely clinical importance"
        else:
            return "represents a large effect of substantial clinical importance"
    elif effect_type == "cohen_f":
        if effect_size < 0.1:
            return "represents a very small effect"
        elif effect_size < 0.25:
            return "represents a small to medium effect"
        elif effect_size < 0.4:
            return "represents a medium to large effect"
        else:
            return "represents a large effect"
    elif effect_type == "hazard_ratio":
        if 0.9 <= effect_size <= 1.1:
            return "represents minimal difference in survival"
        elif effect_size < 0.7 or effect_size > 1.5:
            return "represents a substantial difference in survival with high clinical significance"
        else:
            return "represents a moderate difference in survival of clinical relevance"
    return "should be evaluated for clinical/practical significance in your specific context"


# ==============================================================================
#                   CLUSTER-RANDOMIZED TRIAL FUNCTIONS
# ==============================================================================
def calculate_design_effect(icc: float, cluster_size: float) -> float:
    """
    Calculate design effect for cluster-randomized trials.

    Design Effect (DEFF) = 1 + (m - 1) * ICC
    where m is the average cluster size and ICC is the intra-cluster correlation.

    Parameters:
    -----------
    icc : float
        Intra-cluster correlation coefficient (0 to 1)
    cluster_size : float
        Average number of participants per cluster

    Returns:
    --------
    float : Design effect (≥ 1)
    """
    if icc < 0 or icc > 1:
        raise ValueError("ICC must be between 0 and 1")
    if cluster_size < 1:
        raise ValueError("Cluster size must be at least 1")

    return 1 + (cluster_size - 1) * icc


def calculate_clusters_needed(individual_n: float, cluster_size: float, icc: float) -> tuple:
    """
    Calculate number of clusters needed for cluster-randomized trial.

    Parameters:
    -----------
    individual_n : float
        Sample size needed if individuals were randomized
    cluster_size : float
        Average number of participants per cluster
    icc : float
        Intra-cluster correlation coefficient

    Returns:
    --------
    tuple : (total_participants_needed, number_of_clusters, design_effect)
    """
    deff = calculate_design_effect(icc, cluster_size)
    total_n_cluster = individual_n * deff
    n_clusters = math.ceil(total_n_cluster / cluster_size)

    return (int(math.ceil(total_n_cluster)), n_clusters, deff)


def interpret_icc(icc: float) -> str:
    """Provide interpretation of ICC magnitude."""
    if icc < 0.01:
        return "Very low ICC (<0.01): Minimal clustering effect"
    elif icc < 0.05:
        return "Low ICC (0.01-0.05): Small clustering effect"
    elif icc < 0.10:
        return "Moderate ICC (0.05-0.10): Moderate clustering effect"
    elif icc < 0.20:
        return "High ICC (0.10-0.20): Substantial clustering effect"
    else:
        return "Very High ICC (>0.20): Large clustering effect - design effect will be substantial"


# ==============================================================================
#                   REPEATED MEASURES ANOVA FUNCTIONS
# ==============================================================================
def calculate_repeated_measures_power(n: int, effect_size: float, alpha: float,
                                     num_measurements: int, correlation: float,
                                     alternative: str = "two-sided") -> Optional[float]:
    """
    Calculate power for repeated measures ANOVA (within-subjects design).

    Uses adjustment for correlation between repeated measures.
    Adjusted effect size: f_adj = f * sqrt(1 - ρ)
    Where ρ is the average correlation between measurements.

    Parameters:
    -----------
    n : int
        Number of participants (subjects)
    effect_size : float
        Cohen's f for the within-subjects effect
    alpha : float
        Significance level
    num_measurements : int
        Number of repeated measurements/time points
    correlation : float
        Average correlation between repeated measurements (0 to 1)
    alternative : str
        'two-sided' or 'one-sided'

    Returns:
    --------
    float : Statistical power (0 to 1)
    """
    try:
        if correlation < 0 or correlation > 1:
            st.error("Correlation must be between 0 and 1")
            return None

        # Adjust effect size for correlation
        # Higher correlation = more efficient repeated measures design
        epsilon = 1.0  # Sphericity assumption (can be adjusted)
        f_adj = effect_size

        # Calculate degrees of freedom
        df_effect = (num_measurements - 1) * epsilon
        df_error = (n - 1) * df_effect

        # Use ANOVA power calculator with adjusted parameters
        power_calc = FTestAnovaPower()

        # For repeated measures, effective n is larger due to within-subjects design
        # Reduction factor due to correlation: 1 - ρ
        effective_f = effect_size / np.sqrt(1 - correlation + 0.0001)  # Avoid division by zero

        power = power_calc.solve_power(
            effect_size=effective_f,
            nobs=n * num_measurements,
            alpha=alpha,
            k_groups=num_measurements
        )

        return max(0.0, min(1.0, power))

    except Exception as e:
        st.error(f"Error in repeated measures power calculation: {e}")
        return None


def calculate_repeated_measures_n(effect_size: float, alpha: float, power: float,
                                  num_measurements: int, correlation: float) -> Optional[int]:
    """
    Calculate required number of subjects for repeated measures ANOVA.

    Parameters:
    -----------
    effect_size : float
        Cohen's f for the within-subjects effect
    alpha : float
        Significance level
    power : float
        Desired statistical power
    num_measurements : int
        Number of repeated measurements/time points
    correlation : float
        Average correlation between repeated measurements

    Returns:
    --------
    int : Number of subjects needed
    """
    try:
        if correlation < 0 or correlation > 1:
            st.error("Correlation must be between 0 and 1")
            return None

        # Adjust effect size for correlation
        effective_f = effect_size / np.sqrt(1 - correlation + 0.0001)

        # Use ANOVA power calculator
        power_calc = FTestAnovaPower()

        # Solve for total observations
        total_obs = power_calc.solve_power(
            effect_size=effective_f,
            nobs=None,
            alpha=alpha,
            power=power,
            k_groups=num_measurements
        )

        # Convert to number of subjects
        n_subjects = math.ceil(total_obs / num_measurements)

        return max(3, n_subjects)  # Minimum 3 subjects

    except Exception as e:
        st.error(f"Error in repeated measures sample size calculation: {e}")
        return None


# ==============================================================================
#                   BAYESIAN SAMPLE SIZE FUNCTIONS
# ==============================================================================
def calculate_assurance(n: int, alpha: float, prior_mean: float, prior_sd: float,
                       target_power: float = 0.80, alternative: str = "two-sided") -> Optional[float]:
    """
    Calculate assurance (probability of achieving target power given prior on effect size).

    Assurance is the expected power averaged over the prior distribution of effect sizes.
    It answers: "What's the probability we'll achieve our target power given our
    uncertainty about the true effect size?"

    Parameters:
    -----------
    n : int
        Planned sample size per group
    alpha : float
        Significance level
    prior_mean : float
        Mean of prior distribution on Cohen's d
    prior_sd : float
        Standard deviation of prior distribution on Cohen's d
    target_power : float
        Target power threshold (default 0.80)
    alternative : str
        'two-sided' or 'one-sided'

    Returns:
    --------
    float : Assurance (probability of achieving target power)
    """
    try:
        # Monte Carlo integration over prior distribution
        n_samples = 10000
        effect_sizes = np.random.normal(prior_mean, prior_sd, n_samples)
        effect_sizes = np.abs(effect_sizes)  # Use absolute values

        powers = []
        power_calc = TTestIndPower()

        for es in effect_sizes:
            if es > 0:
                try:
                    pwr = power_calc.solve_power(
                        effect_size=es,
                        nobs1=n,
                        alpha=alpha,
                        ratio=1.0,
                        alternative=alternative
                    )
                    powers.append(pwr if math.isfinite(pwr) else 0)
                except:
                    powers.append(0)
            else:
                powers.append(0)

        # Assurance = P(Power > target_power)
        assurance = np.mean(np.array(powers) >= target_power)

        return assurance

    except Exception as e:
        st.error(f"Error calculating assurance: {e}")
        return None


def calculate_bayesian_sample_size(alpha: float, prior_mean: float, prior_sd: float,
                                   target_assurance: float = 0.80,
                                   target_power: float = 0.80,
                                   alternative: str = "two-sided",
                                   max_n: int = 10000) -> Optional[int]:
    """
    Calculate sample size to achieve target assurance.

    Parameters:
    -----------
    alpha : float
        Significance level
    prior_mean : float
        Mean of prior distribution on Cohen's d
    prior_sd : float
        Standard deviation of prior distribution
    target_assurance : float
        Desired assurance level (default 0.80)
    target_power : float
        Target power threshold for assurance calculation (default 0.80)
    alternative : str
        'two-sided' or 'one-sided'
    max_n : int
        Maximum sample size to search (default 10000)

    Returns:
    --------
    int : Sample size per group to achieve target assurance
    """
    try:
        # Binary search for required sample size
        low, high = 10, max_n
        result_n = None

        while low <= high:
            mid = (low + high) // 2
            assurance = calculate_assurance(mid, alpha, prior_mean, prior_sd,
                                           target_power, alternative)

            if assurance is None:
                return None

            if assurance >= target_assurance:
                result_n = mid
                high = mid - 1  # Try smaller n
            else:
                low = mid + 1   # Need larger n

        return result_n if result_n else max_n

    except Exception as e:
        st.error(f"Error in Bayesian sample size calculation: {e}")
        return None


def calculate_expected_power(n: int, alpha: float, prior_mean: float, prior_sd: float,
                            alternative: str = "two-sided") -> Optional[float]:
    """
    Calculate expected power (average power over prior distribution).

    Parameters:
    -----------
    n : int
        Sample size per group
    alpha : float
        Significance level
    prior_mean : float
        Mean of prior distribution on Cohen's d
    prior_sd : float
        Standard deviation of prior distribution
    alternative : str
        'two-sided' or 'one-sided'

    Returns:
    --------
    float : Expected power (average over prior)
    """
    try:
        # Monte Carlo integration
        n_samples = 10000
        effect_sizes = np.random.normal(prior_mean, prior_sd, n_samples)
        effect_sizes = np.abs(effect_sizes)

        powers = []
        power_calc = TTestIndPower()

        for es in effect_sizes:
            if es > 0:
                try:
                    pwr = power_calc.solve_power(
                        effect_size=es,
                        nobs1=n,
                        alpha=alpha,
                        ratio=1.0,
                        alternative=alternative
                    )
                    powers.append(pwr if math.isfinite(pwr) else 0)
                except:
                    powers.append(0)
            else:
                powers.append(0)

        expected_power = np.mean(powers)

        return expected_power

    except Exception as e:
        st.error(f"Error calculating expected power: {e}")
        return None


# ==============================================================================
#                        STREAMLINED TEST CONFIGURATIONS
# ==============================================================================
def get_test_config(test_name: str) -> Dict:
    """Get test configuration dynamically to reduce memory footprint."""
    base_configs = {
        # Parametric Tests
        "Two-Sample Independent Groups t-test": {
            "key": "2samp", "class": TTestIndPower, "effect": "cohen_d_two",
            "benchmarks": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
            "raw_inputs": ["mean1", "mean2", "pooled_sd"], "n_ratio": True,
            "n_labels": ["Required N₁", "Required N₂", "Total N Required"]
        },
        "One-Sample t-test": {
            "key": "1samp", "class": TTestPower, "effect": "cohen_d_one", "nobs_total": True,
            "benchmarks": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
            "raw_inputs": ["hypothesized_mean", "sample_mean", "sd"],
            "n_labels": ["Required Sample Size (N)"]
        },
        "Paired t-test": {
            "key": "paired", "class": TTestPower, "effect": "cohen_d_paired", "nobs_total": True,
            "benchmarks": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
            "raw_inputs": ["mean_diff", "sd_diff"],
            "n_labels": ["Required Number of Pairs (N)"]
        },
        "Z-test: Two Independent Proportions": {
            "key": "prop", "func": "power_proportions_2indep", "effect": "cohen_h",
            "raw_inputs": ["prop1", "prop2"], "n_ratio": True, "check_counts": "two_prop",
            "n_labels": ["Required N₁", "Required N₂", "Total N Required"]
        },
        "Z-test: Single Proportion": {
            "key": "singleprop", "func": "calculate_single_proportion_power", "effect": "cohen_h",
            "raw_inputs": ["null_prop", "sample_prop"], "check_counts": "one_prop",
            "n_labels": ["Required Sample Size (N)"]
        },
        "One-Way ANOVA (Between Subjects)": {
            "key": "anova", "class": FTestAnovaPower, "effect": "cohen_f",
            "benchmarks": {"Small": 0.10, "Medium": 0.25, "Large": 0.40},
            "k_groups": True, "fixed_alt": True, "nobs_total": True,
            "n_labels": ["Required N per Group", "Total N Required"]
        },
        # Non-Parametric (Approximations)
        "Mann-Whitney U Test": {
            "key": "mw", "class": TTestIndPower, "effect": "cohen_d_two", "are": "mann_whitney",
            "raw_inputs": ["median1", "median2", "pooled_sd"], "n_ratio": True,
            "n_labels": ["Approx. Required N₁", "Approx. Required N₂", "Approx. Total N"]
        },
        "Wilcoxon Signed-Rank Test": {
            "key": "wilcox", "class": TTestPower, "effect": "wilcoxon_special",
            "are": "wilcoxon", "nobs_total": True,
            "n_labels": ["Approx. Required N"]
        },
        "Kruskal-Wallis Test": {
            "key": "kw", "class": FTestAnovaPower, "effect": "cohen_f", "are": "kruskal_wallis",
            "k_groups": True, "fixed_alt": True, "nobs_total": True,
            "n_labels": ["Approx. N per Group", "Approx. Total N"]
        },
        "Fisher's Exact Test": {
            "key": "fisher", "func": "power_proportions_2indep", "effect": "cohen_h",
            "raw_inputs": ["prop1", "prop2"], "n_ratio": True, "fisher": True,
            "n_labels": ["Approx. Required N₁", "Approx. Required N₂", "Approx. Total N"]
        },
        # Survival Analysis
        "Log-Rank Test": {
            "key": "logrank", "func": "calculate_logrank_power", "effect": "hazard_ratio",
            "raw_inputs": ["hazard_ratio", "prob_event"], "n_ratio": True,
            "n_labels": ["Required N₁", "Required N₂", "Total N Required"],
            "benchmarks": {"Small (HR=0.8)": 0.8, "Medium (HR=0.65)": 0.65, "Large (HR=0.5)": 0.5}
        },
        # Cluster-Randomized Trials
        "Cluster-Randomized t-test": {
            "key": "crt_ttest", "class": TTestIndPower, "effect": "cohen_d_two",
            "benchmarks": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
            "raw_inputs": ["mean1", "mean2", "pooled_sd"], "n_ratio": True,
            "cluster_randomized": True,
            "n_labels": ["Required N₁", "Required N₂", "Total N Required"]
        },
        "Cluster-Randomized Proportion Test": {
            "key": "crt_prop", "func": "power_proportions_2indep", "effect": "cohen_h",
            "raw_inputs": ["prop1", "prop2"], "n_ratio": True,
            "cluster_randomized": True, "check_counts": "two_prop",
            "n_labels": ["Required N₁", "Required N₂", "Total N Required"]
        },
        # Repeated Measures
        "Repeated Measures ANOVA": {
            "key": "rm_anova", "func": "calculate_repeated_measures_power", "effect": "cohen_f",
            "benchmarks": {"Small": 0.10, "Medium": 0.25, "Large": 0.40},
            "repeated_measures": True, "nobs_total": True,
            "n_labels": ["Required N (Subjects)", "Total Observations"]
        },
        # Bayesian Methods
        "Bayesian Sample Size (Assurance)": {
            "key": "bayesian_assurance", "func": "calculate_bayesian_sample_size",
            "effect": "cohen_d_two", "bayesian": True,
            "benchmarks": {"Small": 0.2, "Medium": 0.5, "Large": 0.8},
            "n_labels": ["Required N₁ (Assurance)", "Required N₂", "Total N Required"]
        }
    }

    return base_configs.get(test_name, {})


def power_proportions_2indep(effect_size: float, nobs1: float, alpha: float, ratio: float = 1,
                             alternative: str = 'two-sided') -> float:
    """Calculate power for two independent proportions test."""
    from statsmodels.stats.power import zt_ind_solve_power
    return zt_ind_solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha,
                              ratio=ratio, alternative=alternative)


def calculate_single_proportion_power(alpha: float, alternative: str, power: Optional[float] = None,
                                      nobs1: Optional[float] = None, sample_prop: Optional[float] = None,
                                      null_prop: Optional[float] = None, **kwargs) -> Optional[float]:
    """Manual calculation for single proportion test with improved error handling."""

    # Input validation with detailed error messages
    if sample_prop is None or null_prop is None:
        st.error("Missing proportion values. Please provide both null proportion and sample proportion.")
        return None

    if not (0 < sample_prop < 1 and 0 < null_prop < 1):
        st.error("Proportions must be between 0 and 1 (exclusive).")
        return None

    if sample_prop == null_prop:
        st.error("Sample proportion must be different from null proportion.")
        return None

    try:
        alpha_crit = alpha / 2 if alternative == "two-sided" else alpha
        z_alpha = norm.ppf(1 - alpha_crit)

        if power is None and nobs1 is not None:  # Calculate power
            if nobs1 <= 0:
                st.error("Sample size must be positive.")
                return None

            se_alt = np.sqrt(sample_prop * (1 - sample_prop) / nobs1)
            se_null = np.sqrt(null_prop * (1 - null_prop) / nobs1)

            if se_alt == 0 or se_null == 0:
                st.error("Standard error calculation failed.")
                return None

            if alternative == "two-sided":
                crit_upper = null_prop + z_alpha * se_null
                crit_lower = null_prop - z_alpha * se_null
                z_lower = (crit_lower - sample_prop) / se_alt
                z_upper = (crit_upper - sample_prop) / se_alt
                result = norm.cdf(z_lower) + (1 - norm.cdf(z_upper))
            else:
                crit = null_prop + (z_alpha if alternative == "larger" else -z_alpha) * se_null
                z_crit = (crit - sample_prop) / se_alt
                result = 1 - norm.cdf(z_crit) if alternative == "larger" else norm.cdf(z_crit)

            return max(0.0, min(1.0, result))

        elif nobs1 is None and power:  # Calculate N
            if not 0 < power < 1:
                st.error(f"Power must be between 0 and 1, got {power}")
                return None

            z_beta = norm.ppf(power)
            num = z_alpha * np.sqrt(null_prop * (1 - null_prop)) + z_beta * np.sqrt(sample_prop * (1 - sample_prop))
            den = abs(sample_prop - null_prop)

            if den == 0:
                st.error("Denominator is zero - proportions are equal.")
                return None

            result = (num / den) ** 2
            return max(1, result)  # Ensure minimum sample size of 1

    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

    return None


def calculate_logrank_power(alpha: float, alternative: str, power: Optional[float] = None,
                            nobs1: Optional[float] = None, hazard_ratio: Optional[float] = None,
                            prob_event: float = 0.5, ratio: float = 1.0, **kwargs) -> Optional[float]:
    """
    Calculate power or sample size for log-rank test using Schoenfeld's formula.

    Parameters:
    -----------
    alpha : float
        Significance level
    alternative : str
        'two-sided', 'larger', or 'smaller'
    power : float, optional
        Desired statistical power (for sample size calculation)
    nobs1 : float, optional
        Sample size in group 1 (for power calculation)
    hazard_ratio : float
        Expected hazard ratio (HR); HR > 1 means worse survival in group 1
    prob_event : float
        Probability of observing an event (default 0.5)
    ratio : float
        Sample size ratio N2/N1 (default 1.0 for equal groups)

    Returns:
    --------
    float : Calculated power or sample size (N1)

    Notes:
    ------
    Based on Schoenfeld's formula for log-rank test sample size.
    Calculates the required number of participants, accounting for event rate.
    """

    # Input validation
    if hazard_ratio is None or hazard_ratio <= 0:
        st.error("Hazard ratio must be positive.")
        return None

    if hazard_ratio == 1:
        st.error("Hazard ratio must be different from 1 (no effect).")
        return None

    if not (0 < prob_event <= 1):
        st.error("Probability of event must be between 0 and 1.")
        return None

    try:
        # Critical values
        alpha_crit = alpha / 2 if alternative == "two-sided" else alpha
        z_alpha = norm.ppf(1 - alpha_crit)

        # Log hazard ratio
        theta = np.log(hazard_ratio)

        if power is None and nobs1 is not None:  # Calculate power
            if nobs1 <= 0:
                st.error("Sample size must be positive.")
                return None

            # Total sample size
            n2 = nobs1 * ratio
            n_total = nobs1 + n2

            # Expected number of events
            d = n_total * prob_event

            # Proportion in each group
            p1 = nobs1 / n_total
            p2 = n2 / n_total

            # Variance of log(HR) estimator
            var_theta = 1 / (d * p1 * p2)

            # Calculate power
            # Test statistic under alternative: z_obs ~ N(theta/sqrt(var), 1)
            z_obs = abs(theta) / np.sqrt(var_theta)

            if alternative == "two-sided":
                # Power = P(|Z| > z_alpha | theta != 0)
                # = Φ(z_obs - z_alpha) + Φ(-z_obs - z_alpha)
                result = 1 - norm.cdf(z_alpha - z_obs) + norm.cdf(-z_alpha - z_obs)
            else:
                # One-sided test
                if alternative == "larger":
                    # HR < 1, theta < 0
                    z_obs = theta / np.sqrt(var_theta)  # Keep negative sign
                    result = norm.cdf(z_obs - z_alpha)
                else:
                    # HR > 1, theta > 0
                    z_obs = theta / np.sqrt(var_theta)
                    result = 1 - norm.cdf(z_alpha - z_obs)

            return max(0.0, min(1.0, result))

        elif nobs1 is None and power is not None:  # Calculate N
            if not 0 < power < 1:
                st.error(f"Power must be between 0 and 1, got {power}")
                return None

            z_beta = norm.ppf(power)

            # Number of events needed (Schoenfeld's formula)
            # For two groups with allocation ratio r = n2/n1:
            # d = (z_alpha + z_beta)² × (1 + r)² / (r × theta²)
            # For equal allocation (r=1): d = 4 × (z_alpha + z_beta)² / theta²
            d_needed = ((z_alpha + z_beta) ** 2) * ((1 + ratio) ** 2) / (ratio * (theta ** 2))

            # Convert events to total sample size
            n_total = d_needed / prob_event

            # Calculate N1 from total N and ratio
            # n_total = n1 + n2 = n1 + n1*ratio = n1(1 + ratio)
            n1 = n_total / (1 + ratio)

            return max(1, n1)

    except Exception as e:
        st.error(f"Calculation error in log-rank test: {str(e)}")
        return None

    return None


# ==============================================================================
#                           MAIN CALCULATION ENGINE
# ==============================================================================
def run_test_calculation(test_name: str):
    """Unified calculation function for all tests."""
    config = get_test_config(test_name)
    if not config:
        st.error(f"Test configuration not found: {test_name}")
        return

    key = config["key"]
    is_approx = bool(config.get("are") or config.get("fisher"))

    # Header
    st.header(f"Power/Sample Size: {test_name}{' (Approximation)' if is_approx else ''}")

    # Show test description
    show_test_descriptions(test_name, config)

    # Sidebar setup
    st.sidebar.header(f"Parameters ({test_name})")
    if st.sidebar.button("Reset Inputs", key=f"reset_{key}"):
        for k in list(st.session_state.keys()):
            if k.endswith(f"_{key}"):
                del st.session_state[k]
        st.rerun()

    # Get inputs
    inputs = collect_inputs(config, key)

    # Calculate
    st.divider()
    st.header(f"Results ({test_name}{' - Approximation' if is_approx else ''})")

    result = perform_calculation(config, inputs)

    if result is not None and math.isfinite(result):
        display_results(config, inputs, result)
    else:
        st.error("Calculation failed. Please check your inputs and try again.")


def collect_inputs(config: Dict, key: str) -> Dict:
    """Collect all inputs from sidebar."""
    inputs = {}

    # Calculation goal
    goal_options = ["Sample Size", "Power", "MDES"]
    if config.get("func") in ["power_proportions_2indep", "calculate_single_proportion_power"]:
        goal_options = goal_options[:2]  # No MDES for proportions
    inputs["goal"] = st.sidebar.radio("1. Calculate?", goal_options, key=f"goal_{key}")

    # Common inputs
    inputs["alpha"] = st.sidebar.slider("2. Significance Level (α)", 0.001, 0.20, 0.05, 0.005, key=f"alpha_{key}")

    if inputs["goal"] != "Power":
        inputs["power"] = st.sidebar.slider("3. Desired Power (1-β)", 0.50, 0.99, 0.80, 0.01, key=f"power_{key}")

    # Study objective
    inputs["objective"] = st.sidebar.selectbox(
        "4. Objective",
        ["Superiority", "Non-Inferiority", "Equivalence"],
        key=f"obj_{key}"
    )

    # Alternative hypothesis
    if config.get("fixed_alt"):
        inputs["alternative"] = "two-sided"
    else:
        alt_map = {"Two-sided": "two-sided", "One-sided (larger)": "larger", "One-sided (smaller)": "smaller"}
        alt_choice = st.sidebar.selectbox("5. Alternative Hypothesis", list(alt_map.keys()), key=f"alt_{key}")
        inputs["alternative"] = alt_map[alt_choice]

        if inputs["objective"] == "Equivalence":
            inputs["alternative"] = "two-sided"
        elif inputs["objective"] == "Non-Inferiority" and inputs["alternative"] == "two-sided":
            inputs["alternative"] = "larger"

    # Effect size (if not calculating MDES)
    if inputs["goal"] != "MDES":
        effect_inputs = collect_effect_size_inputs(config, key)
        inputs.update(effect_inputs)

    # k groups for ANOVA/KW
    if config.get("k_groups"):
        inputs["k_groups"] = st.sidebar.number_input("6. Number of Groups (k)", min_value=3, value=3, key=f"k_{key}")

    # Sample size inputs
    if config.get("n_ratio"):
        inputs["n_ratio"] = st.sidebar.number_input("7. Sample Size Ratio (N₂/N₁)", 0.1, 10.0, 1.0, 0.1,
                                                    key=f"ratio_{key}")

    if inputs["goal"] != "Sample Size":
        n_label = "Sample Size (N)" if config.get("nobs_total") else "Sample Size Group 1 (N₁)"
        inputs["n"] = st.sidebar.number_input(f"8. {n_label}", min_value=3, value=30, key=f"n_{key}")

    # Dropout
    if inputs["goal"] == "Sample Size":
        inputs["dropout"] = st.sidebar.slider("Dropout Rate (%)", 0, 50, 0, 1, key=f"dropout_{key}")

    # Cluster-randomized parameters
    if config.get("cluster_randomized"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Cluster-Randomized Design Parameters**")

        inputs["cluster_size"] = st.sidebar.number_input(
            "Average Cluster Size (m)",
            min_value=2,
            value=20,
            key=f"cluster_size_{key}",
            help="Average number of participants per cluster (e.g., patients per clinic, students per school)"
        )

        # ICC input with guidance
        icc_default = 0.05
        inputs["icc"] = st.sidebar.slider(
            "Intra-Cluster Correlation (ICC)",
            min_value=0.001,
            max_value=0.50,
            value=icc_default,
            step=0.001,
            format="%.3f",
            key=f"icc_{key}",
            help="Correlation between observations within the same cluster. Typical ranges: Primary care 0.01-0.05, Schools 0.05-0.20"
        )

        st.sidebar.info(interpret_icc(inputs["icc"]))

    # Repeated measures parameters
    if config.get("repeated_measures"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔁 Repeated Measures Parameters**")

        inputs["num_measurements"] = st.sidebar.number_input(
            "Number of Measurements",
            min_value=2,
            value=3,
            key=f"num_meas_{key}",
            help="Number of repeated measurements/time points per subject"
        )

        inputs["correlation"] = st.sidebar.slider(
            "Avg. Correlation Between Measures",
            min_value=0.0,
            max_value=0.95,
            value=0.5,
            step=0.05,
            key=f"corr_{key}",
            help="Average correlation between repeated measurements. Higher correlation = more efficient design."
        )

    # Bayesian parameters
    if config.get("bayesian"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎲 Bayesian Prior on Effect Size**")

        prior_choice = st.sidebar.selectbox(
            "Prior Distribution",
            ["Skeptical", "Neutral", "Optimistic", "Enthusiastic", "Custom"],
            key=f"prior_{key}",
            help="Prior belief about effect size distribution"
        )

        if prior_choice == "Custom":
            inputs["prior_mean"] = st.sidebar.number_input(
                "Prior Mean (Cohen's d)",
                min_value=0.0,
                value=0.3,
                step=0.05,
                key=f"prior_mean_{key}"
            )
            inputs["prior_sd"] = st.sidebar.number_input(
                "Prior SD",
                min_value=0.05,
                value=0.25,
                step=0.05,
                key=f"prior_sd_{key}"
            )
        else:
            prior_params = BAYESIAN_PRIORS[prior_choice.lower()]
            inputs["prior_mean"] = prior_params["mean"]
            inputs["prior_sd"] = prior_params["sd"]

        st.sidebar.info(f"Prior: N({inputs['prior_mean']:.2f}, {inputs['prior_sd']:.2f})")

        if inputs["goal"] == "Sample Size":
            inputs["target_assurance"] = st.sidebar.slider(
                "Target Assurance",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
                key=f"assurance_{key}",
                help="Probability of achieving target power given prior uncertainty"
            )
            inputs["target_power"] = st.sidebar.slider(
                "Target Power Threshold",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
                key=f"target_power_{key}",
                help="Power threshold for assurance calculation"
            )

    return inputs


def collect_effect_size_inputs(config: Dict, key: str) -> Dict:
    """Collect effect size inputs."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**4. Effect Size**")

    effect_inputs = {}
    effect_type = config.get("effect", "")

    # Special handling for Wilcoxon
    if effect_type == "wilcoxon_special":
        return handle_wilcoxon_effect_inputs(key)

    # Special handling for hazard ratio (survival analysis)
    if effect_type == "hazard_ratio":
        return handle_hazard_ratio_inputs(config, key)

    # Check if raw inputs available
    raw_available = bool(config.get("raw_inputs"))

    # For single proportion test, force raw values since we need actual proportions
    if config.get("key") == "singleprop":
        method = "Raw Values"
        st.sidebar.info("ℹ️ Single proportion tests require actual proportion values")
    elif raw_available:
        method = st.sidebar.radio("Specify Effect via:", ["Standardized", "Raw Values"], key=f"method_{key}")
    else:
        method = "Standardized"

    if method == "Standardized":
        benchmarks = config.get("benchmarks", {"Small": 0.2, "Medium": 0.5, "Large": 0.8})
        preset_opts = [f"{k} ({v:.2f})" for k, v in benchmarks.items()] + ["Custom"]
        preset = st.sidebar.radio("Select effect size:", preset_opts, index=1, key=f"preset_{key}")

        if preset == "Custom":
            effect_inputs["effect_size"] = st.sidebar.number_input("Custom Effect Size", 0.001, value=0.5, step=0.01,
                                                                   key=f"custom_{key}")
        else:
            effect_inputs["effect_size"] = float(preset.split("(")[1].split(")")[0])
    else:
        # Raw value inputs
        raw_vals = {}
        raw_map = {
            "mean1": ("Mean Group 1 (μ₁)", 0.0),
            "mean2": ("Mean Group 2 (μ₂)", 1.0),
            "pooled_sd": ("Pooled SD (σ)", 1.0),
            "median1": ("Median Group 1", 0.0),
            "median2": ("Median Group 2", 1.0),
            "hypothesized_mean": ("Hypothesized Mean (μ₀)", 0.0),
            "sample_mean": ("Expected Sample Mean (μ)", 1.0),
            "sd": ("Standard Deviation (σ)", 1.0),
            "mean_diff": ("Mean of Differences", 0.5),
            "sd_diff": ("SD of Differences", 1.0),
            "prop1": ("Proportion Group 1", 0.25),
            "prop2": ("Proportion Group 2", 0.40),
            "null_prop": ("Null Proportion (p₀)", 0.50),
            "sample_prop": ("Expected Proportion (p)", 0.60)
        }

        for input_name in config.get("raw_inputs", []):
            label, default = raw_map.get(input_name, (input_name, 0.0))
            if "prop" in input_name:
                raw_vals[input_name] = st.sidebar.slider(
                    label, 0.001, 0.999, default, 0.005,
                    key=f"{input_name}_{key}",
                    help=f"Enter the {label.lower()} as a decimal between 0 and 1"
                )
            else:
                min_val = 0.0001 if "sd" in input_name else None
                raw_vals[input_name] = st.sidebar.number_input(
                    label, value=default, min_value=min_val,
                    key=f"{input_name}_{key}",
                    help=f"Enter the {label.lower()}"
                )

        # Calculate effect size
        if effect_type == "cohen_h":
            # For single proportion, use sample_prop vs null_prop
            if "sample_prop" in raw_vals and "null_prop" in raw_vals:
                effect_inputs["effect_size"] = calculate_effect_size(effect_type,
                                                                     p1=raw_vals["sample_prop"],
                                                                     p2=raw_vals["null_prop"])
            # For two proportions, use prop1 vs prop2
            elif "prop1" in raw_vals and "prop2" in raw_vals:
                effect_inputs["effect_size"] = calculate_effect_size(effect_type,
                                                                     p1=raw_vals["prop1"],
                                                                     p2=raw_vals["prop2"])
        else:
            effect_inputs["effect_size"] = calculate_effect_size(effect_type, **raw_vals)

        # Always store raw_vals for proportion tests
        effect_inputs["raw_vals"] = raw_vals

        if effect_inputs.get("effect_size"):
            st.sidebar.info(f"Calculated Effect Size = {effect_inputs['effect_size']:.3f}")
        elif config.get("key") == "singleprop":
            # For single proportion, we don't necessarily need the effect size to be calculated
            # since we pass the raw proportions directly to the calculation
            st.sidebar.info("Raw proportions will be used directly in calculation")
        else:
            st.sidebar.error("Could not calculate effect size. Please check your inputs.")

    return effect_inputs


def handle_wilcoxon_effect_inputs(key: str) -> Dict:
    """Special handler for Wilcoxon effect size inputs."""
    method = st.sidebar.radio("Specify Effect via:", ["Standardized (d/d_z)", "Raw Values"], key=f"method_{key}")
    effect_inputs = {}

    if method == "Standardized (d/d_z)":
        preset_opts = ["Small (0.20)", "Medium (0.50)", "Large (0.80)", "Custom"]
        preset = st.sidebar.radio("Select d/d_z:", preset_opts, index=1, key=f"preset_{key}")

        if preset == "Custom":
            effect_inputs["effect_size"] = st.sidebar.number_input("Custom d/d_z", 0.001, value=0.5, step=0.01,
                                                                   key=f"custom_{key}")
        else:
            effect_inputs["effect_size"] = float(preset.split("(")[1].split(")")[0])
    else:
        is_paired = st.sidebar.checkbox("Paired Data?", value=True, key=f"paired_{key}")

        if is_paired:
            median_diff = st.sidebar.number_input("Median of Differences", value=0.5, key=f"mdiff_{key}")
            sd_diff = st.sidebar.number_input("SD of Differences", min_value=0.0001, value=1.0, key=f"sddiff_{key}")
            effect_inputs["effect_size"] = abs(median_diff) / sd_diff if sd_diff > 0 else None
        else:
            null_median = st.sidebar.number_input("Null Median", value=0.0, key=f"null_{key}")
            sample_median = st.sidebar.number_input("Sample Median", value=1.0, key=f"samp_{key}")
            sd = st.sidebar.number_input("Standard Deviation", min_value=0.0001, value=1.0, key=f"sd_{key}")
            effect_inputs["effect_size"] = abs(sample_median - null_median) / sd if sd > 0 else None

        effect_inputs["is_paired"] = is_paired
        if effect_inputs["effect_size"]:
            st.sidebar.info(f"Approx. d/d_z = {effect_inputs['effect_size']:.3f}")

    return effect_inputs


def handle_hazard_ratio_inputs(config: Dict, key: str) -> Dict:
    """Special handler for hazard ratio inputs (survival analysis)."""
    effect_inputs = {}

    st.sidebar.info("ℹ️ Hazard Ratio (HR): ratio of hazard rates between groups. HR=1 means no effect.")

    # Hazard ratio input
    benchmarks = config.get("benchmarks", {"Small (HR=0.8)": 0.8, "Medium (HR=0.65)": 0.65, "Large (HR=0.5)": 0.5})
    preset_opts = [f"{k}" for k in benchmarks.keys()] + ["Custom"]
    preset = st.sidebar.radio("Select Hazard Ratio:", preset_opts, index=1, key=f"preset_hr_{key}")

    if preset == "Custom":
        hr_val = st.sidebar.number_input(
            "Hazard Ratio (HR)",
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.05,
            key=f"custom_hr_{key}",
            help="HR < 1: reduced hazard (better survival) in Group 1. HR > 1: increased hazard (worse survival) in Group 1."
        )
    else:
        # Extract HR value from preset
        hr_val = benchmarks[preset]

    effect_inputs["hazard_ratio"] = hr_val
    effect_inputs["effect_size"] = hr_val  # Store as effect_size for compatibility

    # Probability of event
    prob_event = st.sidebar.slider(
        "Probability of Event (per group)",
        min_value=0.05,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"prob_event_{key}",
        help="Expected proportion of participants who will experience the event during study follow-up."
    )
    effect_inputs["prob_event"] = prob_event

    # Display interpretation
    if hr_val < 1:
        st.sidebar.success(f"HR = {hr_val:.2f}: {((1-hr_val)*100):.1f}% hazard reduction in Group 1")
    elif hr_val > 1:
        st.sidebar.warning(f"HR = {hr_val:.2f}: {((hr_val-1)*100):.1f}% hazard increase in Group 1")
    else:
        st.sidebar.info(f"HR = {hr_val:.2f}: No difference between groups")

    return effect_inputs


def perform_calculation(config: Dict, inputs: Dict) -> Optional[float]:
    """Perform the power calculation with improved error handling."""
    goal = inputs["goal"]
    effect = inputs.get("effect_size")
    n = inputs.get("n")
    power = inputs.get("power")
    alpha = inputs["alpha"]
    alt = inputs["alternative"]

    # Enhanced validation with specific error messages
    if goal == "Sample Size":
        # For single proportion, we need raw proportions, not necessarily effect size
        if config.get("key") == "singleprop":
            raw_vals = inputs.get("raw_vals", {})
            if not raw_vals.get("null_prop") or not raw_vals.get("sample_prop"):
                st.warning("Please provide valid null proportion and sample proportion values.")
                return None
        else:
            if not effect or effect <= 0:
                st.warning("Please provide a valid effect size (must be > 0).")
                return None
        if not power or not (0 < power < 1):
            st.warning("Please provide a valid power value (between 0 and 1).")
            return None
    elif goal == "Power":
        # For single proportion, we need raw proportions, not necessarily effect size
        if config.get("key") == "singleprop":
            raw_vals = inputs.get("raw_vals", {})
            if not raw_vals.get("null_prop") or not raw_vals.get("sample_prop"):
                st.warning("Please provide valid null proportion and sample proportion values.")
                return None
        else:
            if effect is None or effect <= 0:
                st.warning("Please provide a valid effect size (must be > 0).")
                return None
        if not n or n < 3:
            st.warning("Please provide a valid sample size (must be ≥ 3).")
            return None
    elif goal == "MDES":
        if not power or not (0 < power < 1):
            st.warning("Please provide a valid power value (between 0 and 1).")
            return None
        if not n or n < 3:
            st.warning("Please provide a valid sample size (must be ≥ 3).")
            return None

    # Check expected counts for proportions
    if config.get("check_counts") and inputs.get("raw_vals"):
        check_expected_counts(config["check_counts"], n or 30, inputs.get("n_ratio", 1.0), inputs["raw_vals"])

    # Apply ARE adjustment
    are_factor = ARE_FACTORS.get(config.get("are"))
    if are_factor:
        if goal in ["Power", "MDES"] and effect is not None:
            effect *= np.sqrt(are_factor)

    # Prepare calculation arguments
    try:
        if config.get("class"):  # Statsmodels class
            calc_class = config["class"]()
            args = {
                "effect_size": effect if goal != "MDES" else None,
                "alpha": alpha,
                "power": power if goal != "Power" else None,
                "alternative": alt if not config.get("fixed_alt") else None
            }

            if config.get("nobs_total"):
                args["nobs"] = n if goal != "Sample Size" else None
            else:
                args["nobs1"] = n if goal != "Sample Size" else None
                args["ratio"] = inputs.get("n_ratio", 1.0)

            if config.get("k_groups"):
                args["k_groups"] = inputs.get("k_groups", 3)

            args = {k: v for k, v in args.items() if
                    v is not None or (goal == "Sample Size" and k in ["nobs", "nobs1"]) or (
                            goal == "Power" and k == "power") or (goal == "MDES" and k == "effect_size")}

            result = calc_class.solve_power(**args)

        elif config.get("func"):  # Direct function
            func_name = config["func"]

            # Repeated Measures ANOVA
            if func_name == "calculate_repeated_measures_power":
                num_measurements = inputs.get("num_measurements", 3)
                correlation = inputs.get("correlation", 0.5)

                if goal == "Sample Size":
                    result = calculate_repeated_measures_n(
                        effect_size=effect,
                        alpha=alpha,
                        power=power,
                        num_measurements=num_measurements,
                        correlation=correlation
                    )
                elif goal == "Power":
                    result = calculate_repeated_measures_power(
                        n=n,
                        effect_size=effect,
                        alpha=alpha,
                        num_measurements=num_measurements,
                        correlation=correlation,
                        alternative=alt
                    )
                else:  # MDES - not directly supported, return None
                    st.warning("MDES calculation not yet supported for Repeated Measures ANOVA.")
                    return None

            # Bayesian Sample Size
            elif func_name == "calculate_bayesian_sample_size":
                prior_mean = inputs.get("prior_mean", 0.3)
                prior_sd = inputs.get("prior_sd", 0.25)
                target_assurance = inputs.get("target_assurance", 0.80)
                target_power = inputs.get("target_power", 0.80)

                if goal == "Sample Size":
                    result = calculate_bayesian_sample_size(
                        alpha=alpha,
                        prior_mean=prior_mean,
                        prior_sd=prior_sd,
                        target_assurance=target_assurance,
                        target_power=target_power,
                        alternative=alt
                    )
                    if result:
                        # Also calculate expected power and assurance for display
                        inputs["expected_power"] = calculate_expected_power(
                            n=int(result),
                            alpha=alpha,
                            prior_mean=prior_mean,
                            prior_sd=prior_sd,
                            alternative=alt
                        )
                        inputs["achieved_assurance"] = calculate_assurance(
                            n=int(result),
                            alpha=alpha,
                            prior_mean=prior_mean,
                            prior_sd=prior_sd,
                            target_power=target_power,
                            alternative=alt
                        )
                elif goal == "Power":
                    # Calculate expected power
                    result = calculate_expected_power(
                        n=n,
                        alpha=alpha,
                        prior_mean=prior_mean,
                        prior_sd=prior_sd,
                        alternative=alt
                    )
                    # Also calculate assurance
                    inputs["assurance"] = calculate_assurance(
                        n=n,
                        alpha=alpha,
                        prior_mean=prior_mean,
                        prior_sd=prior_sd,
                        target_power=0.80,
                        alternative=alt
                    )
                else:  # MDES not applicable for Bayesian
                    st.warning("MDES calculation not applicable for Bayesian methods.")
                    return None

            elif func_name == "calculate_single_proportion_power":
                raw_vals = inputs.get("raw_vals", {})
                result = calculate_single_proportion_power(
                    alpha=alpha,
                    alternative=alt,
                    power=power if goal != "Power" else None,
                    nobs1=n if goal != "Sample Size" else None,
                    sample_prop=raw_vals.get("sample_prop"),
                    null_prop=raw_vals.get("null_prop")
                )
            elif func_name == "calculate_logrank_power":
                hazard_ratio = inputs.get("hazard_ratio")
                prob_event = inputs.get("prob_event", 0.5)
                result = calculate_logrank_power(
                    alpha=alpha,
                    alternative=alt,
                    power=power if goal != "Power" else None,
                    nobs1=n if goal != "Sample Size" else None,
                    hazard_ratio=hazard_ratio,
                    prob_event=prob_event,
                    ratio=inputs.get("n_ratio", 1.0)
                )
            else:  # Two proportions
                if goal == "Sample Size":
                    from statsmodels.stats.power import zt_ind_solve_power
                    result = zt_ind_solve_power(effect_size=effect, nobs1=None, alpha=alpha,
                                                power=power, ratio=inputs.get("n_ratio", 1.0),
                                                alternative=alt)
                elif goal == "Power":
                    result = power_proportions_2indep(effect_size=effect, nobs1=n, alpha=alpha,
                                                      ratio=inputs.get("n_ratio", 1.0), alternative=alt)
                else:  # MDES not supported for proportions
                    return None
        else:
            st.error("No calculation method defined for this test.")
            return None

    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

    # Apply post-calculation adjustments
    if result is not None and math.isfinite(result):
        if are_factor:
            if goal == "Sample Size":
                result /= are_factor
            elif goal == "MDES":
                result /= np.sqrt(are_factor)

        if config.get("fisher"):
            if goal == "Sample Size":
                result *= FISHER_ADJUSTMENTS["n"]
            elif goal == "Power":
                result *= FISHER_ADJUSTMENTS["power"]

        # Cluster-randomized adjustment
        if config.get("cluster_randomized") and goal == "Sample Size":
            cluster_size = inputs.get("cluster_size", 20)
            icc = inputs.get("icc", 0.05)

            # Calculate design effect and adjust sample size
            total_n, n_clusters, deff = calculate_clusters_needed(result, cluster_size, icc)

            # Store cluster information for display
            inputs["individual_n"] = int(math.ceil(result))
            inputs["cluster_adjusted_n"] = total_n
            inputs["n_clusters"] = n_clusters
            inputs["design_effect"] = deff

            # Return individual-level sample size (display will show cluster details)
            result = result  # Keep as individual n for now, display handles cluster info

        if goal == "Power":
            result = max(0.0, min(1.0, result))

    return result


def display_results(config: Dict, inputs: Dict, result: float):
    """Display calculation results with enhanced guidance and practical context."""
    goal = inputs["goal"]
    test_name = st.session_state.get("selected_test", "Statistical Test")

    st.subheader("Calculated Result:")

    if goal == "Sample Size":
        n1 = int(np.ceil(result))
        n_ratio = inputs.get("n_ratio", 1.0)
        n2 = int(np.ceil(n1 * n_ratio)) if config.get("n_ratio") else None
        k = inputs.get("k_groups", 1 if not n2 else 2)
        total = n1 * k if not n2 else n1 + n2

        cols = st.columns(len([x for x in [n1, n2, total] if x]))
        labels = config.get("n_labels", ["N Required"])
        vals = [n1]
        if n2: vals.append(n2)
        if len(labels) > len(vals): vals.append(total)

        for i, (col, label, val) in enumerate(zip(cols, labels, vals)):
            with col:
                st.metric(label, f"{val:d}")

        # Dropout adjustment
        dropout = inputs.get("dropout", 0)
        if dropout > 0:
            st.markdown("---")
            st.subheader(f"Adjusted for {dropout}% Dropout:")
            n1_adj = int(np.ceil(n1 / (1 - dropout / 100)))
            n2_adj = int(np.ceil(n1_adj * n_ratio)) if n2 else None
            total_adj = n1_adj * k if not n2 else n1_adj + (n2_adj or 0)

            adj_vals = [n1_adj]
            if n2_adj: adj_vals.append(n2_adj)
            if len(labels) > len(adj_vals): adj_vals.append(total_adj)

            adj_cols = st.columns(len(adj_vals))
            for i, (col, label, val, orig) in enumerate(zip(adj_cols, labels, adj_vals, vals)):
                with col:
                    st.metric(f"Adj. {label}", f"{val:d}", delta=f"{val - orig:d} increase")

            final_total = total_adj
        else:
            final_total = total

        # Cluster-Randomized Results
        if config.get("cluster_randomized") and inputs.get("design_effect"):
            st.markdown("---")
            st.subheader("📊 Cluster-Randomized Trial Results")

            deff = inputs["design_effect"]
            n_clusters = inputs["n_clusters"]
            cluster_adj_n = inputs["cluster_adjusted_n"]
            individual_n = inputs["individual_n"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Individual-Level N", f"{individual_n:d}")
            with col2:
                st.metric("Design Effect (DEFF)", f"{deff:.3f}")
            with col3:
                st.metric("Total N (Cluster-Adjusted)", f"{cluster_adj_n:d}")

            st.info(f"**Number of Clusters Needed:** {n_clusters} clusters ({n_clusters//2} per group if equal allocation)")
            st.warning(f"⚠️ The clustering inflates required sample size by {((deff-1)*100):.1f}%. This accounts for correlation within clusters (ICC={inputs.get('icc', 0.05):.3f}).")

            final_total = cluster_adj_n  # Use cluster-adjusted for feasibility

        # Bayesian Results
        if config.get("bayesian"):
            st.markdown("---")
            st.subheader("🎲 Bayesian Sample Size Results")

            if goal == "Sample Size":
                exp_power = inputs.get("expected_power")
                achieved_assurance = inputs.get("achieved_assurance")

                if exp_power:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Expected Power", f"{exp_power:.3f}")
                    with col2:
                        st.metric("Achieved Assurance", f"{achieved_assurance:.3f}" if achieved_assurance else "N/A")

                    st.info(f"**Expected Power:** Average power over your prior distribution: {exp_power:.1%}")
                    st.info(f"**Assurance:** Probability of achieving ≥{inputs.get('target_power', 0.80):.0%} power: {achieved_assurance:.1%}" if achieved_assurance else "Assurance calculation in progress...")

        # Repeated Measures Results
        if config.get("repeated_measures"):
            st.markdown("---")
            st.subheader("🔁 Repeated Measures ANOVA Results")

            num_meas = inputs.get("num_measurements", 3)
            correlation = inputs.get("correlation", 0.5)
            total_obs = n1 * num_meas

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Subjects Needed", f"{n1:d}")
            with col2:
                st.metric("Total Observations", f"{total_obs:d}")
            with col3:
                st.metric("Measurements/Subject", f"{num_meas:d}")

            st.info(f"✅ **Within-subjects design advantage:** Correlation of {correlation:.2f} between measurements reduces required subjects compared to between-subjects design.")

        # Practical Guidance for Sample Size
        st.markdown("---")
        st.subheader("Practical Considerations")

        # Recruitment feasibility
        feasibility = assess_recruitment_feasibility(final_total, test_name)
        st.info(feasibility)

        # Timeline estimate
        timeline = estimate_study_timeline(final_total)
        st.info(f"⏱️ **{timeline}**")

        # Effect size validation
        effect_size = inputs.get("effect_size")
        if effect_size:
            warnings = validate_effect_size(effect_size, config.get("effect", ""), test_name)
            if warnings:
                for warning in warnings:
                    if "❌" in warning:
                        st.error(warning)
                    elif "⚠️" in warning:
                        st.warning(warning)
                    else:
                        st.info(warning)

        # Sample size justification
        st.markdown("---")
        justification = generate_sample_size_justification(config, inputs, result)
        with st.expander("📄 Sample Size Justification (for protocols)", expanded=False):
            st.markdown(justification)
            st.info("💡 **Tip:** Copy this text for your study protocol or grant application. Modify as needed to fit your specific context.")

    elif goal == "Power":
        st.metric("Calculated Power (1-β)", f"{result:.3f}")
        st.write(f"Probability of detecting the effect: {result:.1%}")

        # Power interpretation
        st.markdown("---")
        st.subheader("Power Interpretation")
        if result < 0.70:
            st.error(f"⚠️ **Low Power ({result:.1%})**: High risk of missing a true effect (Type II error). Consider increasing sample size.")
        elif result < 0.80:
            st.warning(f"⚠️ **Below Conventional Threshold ({result:.1%})**: Most studies aim for ≥80% power. Consider if this is adequate for your study.")
        elif result < 0.90:
            st.success(f"✅ **Adequate Power ({result:.1%})**: Conventional threshold met. Reasonable chance of detecting the effect if it exists.")
        else:
            st.success(f"✅ **High Power ({result:.1%})**: Excellent chance of detecting the effect. Sample may be larger than minimally necessary.")

        # Effect size validation
        effect_size = inputs.get("effect_size")
        if effect_size:
            warnings = validate_effect_size(effect_size, config.get("effect", ""), test_name)
            if warnings:
                st.markdown("---")
                st.subheader("Effect Size Considerations")
                for warning in warnings:
                    if "❌" in warning:
                        st.error(warning)
                    elif "⚠️" in warning:
                        st.warning(warning)
                    else:
                        st.info(warning)

    elif goal == "MDES":
        st.metric("MDES", f"{result:.3f}")
        st.write(f"Smallest detectable standardized effect: {result:.3f}")

        # MDES interpretation
        st.markdown("---")
        st.subheader("MDES Interpretation")
        effect_type = config.get("effect", "")
        interpretation = _interpret_clinical_significance(result, effect_type)
        st.info(f"This effect size {interpretation}")

        # Practical guidance
        if effect_type in ["cohen_d_two", "cohen_d_one", "cohen_d_paired"]:
            if result > 0.8:
                st.warning("⚠️ Your study can only detect large effects. Consider whether smaller, clinically relevant effects might be missed.")
            elif result < 0.2:
                st.success("✅ Your study is sensitive to small effects. This provides good assurance of detecting clinically meaningful differences.")

    # Summary table
    st.markdown("---")
    st.subheader("Summary of Inputs")
    summary = {
        "Parameter": ["Calculation Goal", "Objective", "Alpha (α)", "Alternative"],
        "Value": [goal, inputs.get("objective", "Superiority"), inputs["alpha"], inputs["alternative"]]
    }

    if "effect_size" in inputs:
        summary["Parameter"].append("Effect Size")
        summary["Value"].append(
            f"{inputs['effect_size']:.3f}" if inputs['effect_size'] is not None else "N/A"
        )

    if "n" in inputs:
        summary["Parameter"].append("Sample Size")
        summary["Value"].append(inputs["n"])

    if "power" in inputs:
        summary["Parameter"].append("Power (1-β)")
        summary["Value"].append(inputs["power"])

    display_results_table(summary)

    # Additional practical tips
    st.markdown("---")
    with st.expander("💡 Best Practices & Common Pitfalls", expanded=False):
        st.markdown("""
        ### Best Practices for Power Analysis:

        1. **Base effect sizes on pilot data or literature**: Don't guess. Use published meta-analyses or well-designed pilot studies.

        2. **Be conservative**: If uncertain, use slightly larger sample sizes or lower power calculations.

        3. **Account for dropout**: Always include expected dropout/attrition in your sample size calculations.

        4. **Consider practical constraints**: A statistically ideal sample size that's practically infeasible is useless.

        5. **Plan for interim analyses**: If you plan interim looks, adjust alpha using methods like O'Brien-Fleming or Lan-DeMets.

        6. **Document all assumptions**: Keep detailed records of where effect size estimates came from and all assumptions made.

        ### Common Pitfalls to Avoid:

        - **Post-hoc power analysis**: Don't calculate power after the study is complete. This is generally not informative.

        - **Inflated effect sizes**: Don't use overly optimistic effect sizes just to get a manageable sample size.

        - **Ignoring multiple comparisons**: If testing multiple outcomes, consider adjusting for multiplicity.

        - **Assuming normality without checking**: Verify distributional assumptions or use non-parametric alternatives.

        - **Forgetting about clinical significance**: Statistical significance ≠ clinical importance. Always consider practical relevance.

        ### Regulatory Considerations:

        - **FDA/ICH E9**: Clinical trials should be adequately powered (typically 80-90%) to address primary endpoints.

        - **Justify your choices**: Regulatory agencies expect detailed justification of sample sizes and power calculations.

        - **Pre-specify analyses**: Register your analysis plan before data collection (e.g., clinicaltrials.gov).
        """)


def show_test_descriptions(test_name: str, config: Dict):
    """Show test-specific descriptions and assumptions."""
    descriptions = {
        "Two-Sample Independent Groups t-test": [
            ("Test Description", "Compares means of a continuous outcome between two **independent** groups.",
             "basic_biostats", True),
            ("Assumptions", "<ol><li>Independence</li><li>Normality within groups</li><li>Equal variances", "basic_biostats", False),
            ("Effect Size", "Cohen's d = |μ₁ - μ₂| / σ_pooled. Benchmarks: 0.2(S), 0.5(M), 0.8(L)", "cohen_1988", False)
        ],
        "One-Sample t-test": [
            ("Test Description", "Compares mean from a single group to a hypothesized value.", "basic_biostats", True),
            ("Assumptions", "<ol><li>Independence</li><li>Normality", "basic_biostats", False),
            ("Effect Size", "d = |μ - μ₀| / σ", "cohen_1988", False)
        ],
        "Paired t-test": [
            ("Test Description", "Compares means for related samples (e.g., pre/post).", "basic_biostats", True),
            ("Assumptions", "<ol><li>Paired data</li><li>Independence of pairs</li><li>Normality of differences", "basic_biostats",
             False),
            ("Effect Size", "d_z = |μ_diff| / σ_diff", "cohen_1988", False)
        ],
        "Z-test: Two Independent Proportions": [
            ("Test Description", "Compares proportions between two independent groups.", "basic_biostats", True),
            ("Assumptions", "<ol><li>Independence</li><li>Binary outcome</li><li>Large sample (expected counts > 5)",
             "normal_approx_prop", False)
        ],
        "Z-test: Single Proportion": [
            ("Test Description", "Compares observed proportion to hypothesized value.", "basic_biostats", True),
            ("Assumptions", "<ol><li>Independence</li><li>Binary outcome</li><li>n*p₀ > 5 and n*(1-p₀) > 5", "normal_approx_prop",
             False)
        ],
        "One-Way ANOVA (Between Subjects)": [
            ("Test Description", "Compares means across 3+ independent groups.", "basic_biostats", True),
            ("Assumptions", "<ol><li>Independence</li><li>Normality within groups</li><li>Equal variances", "basic_biostats", False),
            ("Effect Size", "Cohen's f = σ_means / σ_within. Benchmarks: 0.10(S), 0.25(M), 0.40(L)", "cohen_f_info",
             False)
        ],
        "Mann-Whitney U Test": [
            ("Test Description", "Non-parametric alternative to two-sample t-test.", "mann_whitney", True),
            ("Assumptions", "<ol><li>Independence</li><li>Ordinal/continuous data</li><li>Similar shapes for median comparison",
             "mann_whitney", False),
            ("Approximation", f"Uses t-test with ARE ≈ {ARE_FACTORS['mann_whitney']:.3f}", "are_info", False)
        ],
        "Wilcoxon Signed-Rank Test": [
            ("Test Description", "Non-parametric alternative to paired t-test or one-sample t-test.",
             "wilcoxon_signed_rank", True),
            ("Assumptions", "<ol><li>Paired/single sample</li><li>Independence</li><li>Symmetry around median", "wilcoxon_signed_rank",
             False),
            ("Approximation", f"Uses t-test with ARE ≈ {ARE_FACTORS['wilcoxon']:.3f}", "are_info", False)
        ],
        "Kruskal-Wallis Test": [
            ("Test Description", "Non-parametric alternative to one-way ANOVA.", "kruskal_wallis", True),
            ("Assumptions", "<ol><li>Independence</li><li>Ordinal/continuous</li><li>Similar shapes", "kruskal_wallis", False),
            ("Approximation", f"Uses ANOVA with ARE ≈ {ARE_FACTORS['kruskal_wallis']:.3f}", "are_info", False)
        ],
        "Fisher's Exact Test": [
            ("Test Description", "Exact test for 2x2 tables, best for small samples.", "fishers_exact", True),
            ("Assumptions", "<ol><li>Independence</li><li>Binary outcome</li><li>Fixed margins", "fishers_exact", False),
            ("Approximation", "Uses Z-test with heuristic adjustments", "statsmodels_prop", False)
        ],
        "Log-Rank Test": [
            ("Test Description", "Compares survival curves between two groups. Used for time-to-event data (e.g., time to death, disease progression, or recurrence).", "logrank_test", True),
            ("Assumptions", "<ol><li>Independence of observations</li><li>Censoring is non-informative</li><li>Proportional hazards (hazard ratio constant over time)</li><li>Similar censoring patterns between groups", "logrank_test", False),
            ("Effect Size", "Hazard Ratio (HR) = ratio of hazard rates. HR < 1: reduced risk in Group 1. HR > 1: increased risk in Group 1. HR = 1: no difference.<br><b>Benchmarks:</b> HR=0.80 (small, 20% reduction), HR=0.65 (medium, 35% reduction), HR=0.50 (large, 50% reduction).", "hazard_ratio", False),
            ("Sample Size Calculation", "Uses Schoenfeld's formula. Calculates number of participants needed based on expected event rate and hazard ratio. Higher event rates require fewer participants to achieve the same number of events.", "schoenfeld", False),
            ("Event Probability", "The probability of observing an event during follow-up affects sample size. Lower event rates require larger sample sizes to observe sufficient events for adequate power.", "schoenfeld", False)
        ],
        "Cluster-Randomized t-test": [
            ("Test Description", "Two-sample t-test adjusted for cluster randomization. Used when randomization occurs at cluster level (e.g., clinics, schools, communities) rather than individual level.", "donner_klar", True),
            ("Assumptions", "<ol><li>Independence between clusters</li><li>Normality within groups</li><li>Observations correlated within clusters</li></ol>", "donner_klar", False),
            ("Design Effect", "DEFF = 1 + (m-1)×ICC, where m is average cluster size and ICC is intra-cluster correlation. Sample size inflated by DEFF to account for clustering.", "icc_info", False),
            ("ICC Guidance", "Typical ICC ranges: Primary care (0.01-0.05), Schools (0.05-0.20), Hospitals (0.01-0.10), Communities (0.001-0.05). Use pilot data or published values from similar studies.", "icc_info", False),
            ("Practical Considerations", "Requires recruiting entire clusters. Larger ICC and cluster sizes increase required total sample size substantially. Consider trade-offs between number and size of clusters.", "donner_klar", False)
        ],
        "Cluster-Randomized Proportion Test": [
            ("Test Description", "Two-proportion Z-test adjusted for cluster randomization. Compares proportions between groups when randomization is at cluster level.", "donner_klar", True),
            ("Assumptions", "<ol><li>Independence between clusters</li><li>Binary outcome</li><li>Observations correlated within clusters</li><li>Large enough clusters for normal approximation</li></ol>", "donner_klar", False),
            ("Design Effect", "Sample size inflated by DEFF = 1 + (m-1)×ICC. Higher ICC requires larger sample sizes.", "icc_info", False),
            ("Practical Considerations", "Cluster-randomized trials are common in community interventions, school-based studies, and health system research.", "donner_klar", False)
        ],
        "Repeated Measures ANOVA": [
            ("Test Description", "Within-subjects ANOVA for comparing means across multiple time points or conditions measured on the same subjects. More efficient than between-subjects designs when correlation is moderate to high.", "repeated_measures", True),
            ("Assumptions", "<ol><li>Independence of subjects</li><li>Normality within conditions</li><li>Sphericity (equal variances of differences)</li><li>No carryover effects (if applicable)</li></ol>", "repeated_measures", False),
            ("Design Advantage", "Within-subjects design requires fewer subjects than between-subjects when measurements are correlated. Higher correlation = greater efficiency.", "repeated_measures", False),
            ("Correlation Parameter", "Average correlation between repeated measurements. Typical values: 0.3-0.7. Higher correlation means fewer subjects needed. Estimate from pilot data or literature.", "repeated_measures", False),
            ("Practical Considerations", "Subjects must be available for all time points. Consider dropout and missing data. Appropriate for longitudinal studies, learning effects research, crossover designs.", "repeated_measures", False)
        ],
        "Bayesian Sample Size (Assurance)": [
            ("Test Description", "Bayesian approach to sample size determination using assurance. Accounts for uncertainty in effect size by specifying a prior distribution. Calculates sample size to achieve target probability of success.", "bayesian_sample_size", True),
            ("Assurance", "Probability of achieving target power given prior uncertainty about true effect size. Example: 80% assurance means 80% chance of achieving 80% power given your prior beliefs.", "assurance", False),
            ("Prior Distribution", "Specifies belief about effect size before study. Skeptical prior (small effects), Optimistic prior (larger effects), or custom. Should be based on previous research, expert opinion, or pilot data.", "bayesian_sample_size", False),
            ("Expected Power", "Average power over the prior distribution. More realistic than point estimate power when effect size is uncertain.", "bayesian_sample_size", False),
            ("When to Use", "When substantial uncertainty exists about effect size, in early-phase research, when incorporating previous studies, or when conservative planning is needed.", "bayesian_sample_size", False),
            ("Interpretation", "Bayesian methods provide probability statements about achieving success rather than frequentist guarantees. Useful for decision-making under uncertainty.", "assurance", False)
        ]
    }

    for header, content, citation, expanded in descriptions.get(test_name, []):
        display_explanation(header, content, citation, expanded=expanded)


# ==============================================================================
#                           TEST SELECTION GUIDE
# ==============================================================================
def show_test_selection_guide():
    """Interactive test selection guide."""
    st.header("Statistical Test Selection Guide")
    st.info("Follow the steps to identify a suitable test.")

    outcome = st.radio("**1. Outcome Type?**", ["Continuous", "Binary/Categorical", "Time-to-event"])

    if outcome == "Continuous":
        groups = st.radio("**2. How many groups?**", ["One", "Two", "Three or more"])

        if groups == "One":
            dist = st.radio("**3. Normally distributed?**", ["Yes", "No"])
            test = "One-Sample t-test" if dist == "Yes" else "Wilcoxon Signed-Rank Test"
            category = "Parametric Tests" if dist == "Yes" else "Non-Parametric Tests"
        elif groups == "Two":
            paired = st.radio("**3. Paired or independent?**", ["Paired", "Independent"])
            if paired == "Paired":
                dist = st.radio("**4. Differences normally distributed?**", ["Yes", "No"])
                test = "Paired t-test" if dist == "Yes" else "Wilcoxon Signed-Rank Test"
                category = "Parametric Tests" if dist == "Yes" else "Non-Parametric Tests"
            else:
                dist = st.radio("**4. Normal & equal variance?**", ["Yes", "No"])
                test = "Two-Sample Independent Groups t-test" if dist == "Yes" else "Mann-Whitney U Test"
                category = "Parametric Tests" if dist == "Yes" else "Non-Parametric Tests"
        else:
            dist = st.radio("**3. Normal & equal variance?**", ["Yes", "No"])
            test = "One-Way ANOVA (Between Subjects)" if dist == "Yes" else "Kruskal-Wallis Test"
            category = "Parametric Tests" if dist == "Yes" else "Non-Parametric Tests"

    elif outcome == "Binary/Categorical":
        groups = st.radio("**2. How many groups?**", ["One", "Two"])
        if groups == "One":
            test, category = "Z-test: Single Proportion", "Parametric Tests"
        else:
            size = st.radio("**3. Expected sample size?**", ["Large (counts > 5)", "Small (counts < 5)"])
            if size.startswith("Large"):
                test, category = "Z-test: Two Independent Proportions", "Parametric Tests"
            else:
                test, category = "Fisher's Exact Test", "Non-Parametric Tests"

    else:  # Time-to-event
        groups = st.radio("**2. How many groups?**", ["Two"])
        if groups == "Two":
            test, category = "Log-Rank Test", "Survival Analysis"
        else:
            st.warning("Multi-group survival analysis not yet supported.")
            return

    st.success(f"**Recommended: {test}**")

    def _apply_test_selection():
        """Callback to apply the recommended test and exit the guide."""
        st.session_state["selected_test_category"] = category
        st.session_state["selected_test_name"] = test
        st.session_state["guide_selection_made"] = True
        # Hide the guide so the main interface is shown on rerun
        st.session_state["show_guide"] = False

    st.button(
        f"Use {test}",
        type="primary",
        on_click=_apply_test_selection,
    )


# Update the main sidebar section as well:
def show_main_interface():
    """Show the main test selection interface."""
    # Check if user just came from the guide
    if st.session_state.get("guide_selection_made", False):
        # Use the selections from the guide
        category = st.session_state.get("selected_test_category", "Parametric Tests")
        selected_test = st.session_state.get("selected_test_name")
        # Clear the guide selection flag
        st.session_state["guide_selection_made"] = False
        # Set the regular session state keys
        st.session_state["test_category"] = category
        st.session_state["selected_test"] = selected_test
    
    # Test category selection
    categories = ["Parametric Tests", "Non-Parametric Tests", "Survival Analysis",
                  "Cluster-Randomized", "Repeated Measures", "Bayesian Methods"]
    category = st.sidebar.radio("**1. Select Test Category:**", categories,
                                key="test_category",
                                help="Choose based on study design: Standard tests, Cluster-randomized, Repeated measures, or Bayesian approaches.")

    # Test selection
    if category == "Parametric Tests":
        tests = ["Two-Sample Independent Groups t-test", "One-Sample t-test", "Paired t-test",
                 "Z-test: Two Independent Proportions", "Z-test: Single Proportion",
                 "One-Way ANOVA (Between Subjects)"]
    elif category == "Non-Parametric Tests":
        tests = ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis Test",
                 "Fisher's Exact Test"]
    elif category == "Survival Analysis":
        tests = ["Log-Rank Test"]
    elif category == "Cluster-Randomized":
        tests = ["Cluster-Randomized t-test", "Cluster-Randomized Proportion Test"]
    elif category == "Repeated Measures":
        tests = ["Repeated Measures ANOVA"]
    elif category == "Bayesian Methods":
        tests = ["Bayesian Sample Size (Assurance)"]
    else:
        tests = []

    selected_test = st.sidebar.radio("Select Specific Test:", tests, key="selected_test")

    st.sidebar.divider()

    # Run calculation
    if selected_test:
        run_test_calculation(selected_test)
    else:
        st.info("Please select a statistical test from the sidebar to begin.")


# ==============================================================================
#                             MAIN APP
# ==============================================================================
st.set_page_config(page_title="Power Calculator", layout="wide", initial_sidebar_state="expanded")
st.title("Power and Sample Size Calculator")

# About section
with st.expander("📘 About this Calculator & User Guide", expanded=False):
    st.markdown("""
    ### Overview

    This **Power and Sample Size Calculator** assists researchers, particularly in pharmaceutical, medical device, and clinical research,
    in determining appropriate sample sizes or estimating statistical power for various study designs. It provides comprehensive
    guidance suitable for both experienced researchers and those with limited statistical background.

    ### Key Features:

    #### Core Calculations:
    - **Sample Size (N)**: Determine how many participants you need
    - **Statistical Power (1-β)**: Estimate probability of detecting an effect
    - **Minimum Detectable Effect Size (MDES)**: Find the smallest effect your study can reliably detect

    #### Supported Statistical Tests:

    **Parametric Tests** (assume specific distributions):
    - Two-Sample Independent Groups t-test
    - One-Sample t-test
    - Paired t-test (repeated measures)
    - Z-test for Two Independent Proportions
    - Z-test for Single Proportion
    - One-Way ANOVA (3+ groups)

    **Non-Parametric Tests** (fewer distributional assumptions, using ARE-based approximations):
    - Mann-Whitney U Test (Wilcoxon Rank-Sum)
    - Wilcoxon Signed-Rank Test
    - Kruskal-Wallis Test (3+ groups)
    - Fisher's Exact Test (small samples)

    **Survival Analysis** (time-to-event data):
    - Log-Rank Test (Schoenfeld's formula)

    **Cluster-Randomized Trials** (cluster-level randomization):
    - Cluster-Randomized t-test (with ICC adjustment)
    - Cluster-Randomized Proportion Test
    - Automatic design effect calculation

    **Repeated Measures** (within-subjects designs):
    - Repeated Measures ANOVA
    - Accounts for correlation between measurements
    - More efficient than between-subjects when correlation is moderate-high

    **Bayesian Methods** (incorporating prior uncertainty):
    - Bayesian Sample Size (Assurance approach)
    - Expected Power calculation
    - Multiple prior distributions (Skeptical, Neutral, Optimistic, Custom)

    #### Enhanced Guidance Features:
    - **Interactive Test Selection Guide**: Answer simple questions to identify the appropriate test
    - **Effect Size Flexibility**: Input using standardized metrics (Cohen's d, f, h, HR) or raw values (means, SDs, proportions)
    - **Practical Assessments**: Recruitment feasibility, timeline estimates, and study planning guidance
    - **Sample Size Justification**: Auto-generated text for protocols and grant applications
    - **Validation & Warnings**: Real-time checks for assumption violations and unrealistic parameters
    - **Dropout Adjustment**: Account for participant attrition
    - **Interpretive Guidance**: Contextual interpretation of power, effect sizes, and results
    - **Best Practices**: Built-in reminders of common pitfalls and regulatory considerations

    ### How to Use This Calculator:

    #### Step 1: Choose Your Test
    **Option A - Use the Guide (Recommended for beginners):**
    1. Check "Show Test Selection Guide" in the sidebar
    2. Answer questions about your outcome type and study design
    3. Click "Use [Test Name]" to proceed

    **Option B - Select Directly:**
    1. Choose Test Category (Parametric/Non-Parametric/Survival)
    2. Select specific test from the list

    #### Step 2: Set Your Calculation Goal
    - **Calculate Sample Size**: If you know the effect size and want to find required N
    - **Calculate Power**: If you have a fixed sample size and want to know your detection probability
    - **Calculate MDES**: If you have a fixed sample size and want to know the smallest detectable effect

    #### Step 3: Enter Parameters
    **Common Parameters (all tests):**
    - **Significance Level (α)**: Usually 0.05 (5% Type I error rate)
    - **Power (1-β)**: Usually 0.80 or 0.90 (80-90% chance of detecting effect)
    - **Study Objective**: Superiority (prove difference), Non-Inferiority (prove not worse), or Equivalence (prove similar)
    - **Alternative Hypothesis**: Two-sided (most common) or one-sided

    **Effect Size (if calculating Sample Size or Power):**
    - Use **Standardized** values if you know Cohen's d, f, or h from literature
    - Use **Raw Values** if you have means, SDs, or proportions from pilot data
    - The calculator will show benchmark values (Small, Medium, Large) based on Cohen's conventions

    **Test-Specific Parameters:**
    - Sample size ratios (for unequal groups)
    - Number of groups (for ANOVA/Kruskal-Wallis)
    - Event probabilities (for survival analysis)
    - Dropout rates (adjusts sample size upward)

    #### Step 4: Review Results & Guidance
    The calculator provides:
    - **Primary Result**: Required N, achieved power, or MDES
    - **Practical Considerations**: Feasibility assessment and timeline estimates
    - **Effect Size Validation**: Warnings if values seem unrealistic
    - **Sample Size Justification**: Copy-paste text for your protocol
    - **Summary Table**: All inputs used (for reproducibility)
    - **Best Practices**: Reminders and pitfalls to avoid

    ### Understanding Key Concepts:

    #### Statistical Power (1-β)
    The probability of correctly detecting an effect when it truly exists.
    - **80% power**: Standard in most fields (20% risk of missing a true effect)
    - **90% power**: Higher standard for critical studies or rare diseases
    - Low power = high risk of "false negative" (Type II error)

    #### Effect Size
    A **standardized** measure of the magnitude of difference or relationship:
    - **Cohen's d**: Difference between means in standard deviation units
    - **Cohen's f**: Ratio of between-group to within-group variability (ANOVA)
    - **Cohen's h**: Difference between proportions (transformed scale)
    - **Hazard Ratio (HR)**: Ratio of event rates (survival analysis)

    Cohen's Benchmarks (Rules of Thumb):
    - **Small**: d=0.2, f=0.1, h=0.2
    - **Medium**: d=0.5, f=0.25, h=0.5
    - **Large**: d=0.8, f=0.4, h=0.8

    ⚠️ **Important**: These are general guidelines. Effect sizes should be based on:
    1. Pilot study data
    2. Published literature in your field
    3. Smallest clinically meaningful difference
    4. Meta-analytic estimates

    #### Significance Level (α)
    The probability of finding an effect when none exists (Type I error/"false positive").
    - **α = 0.05**: Standard in most fields (5% risk)
    - **α = 0.01**: More stringent (1% risk, often used with multiple comparisons)
    - Lower α requires larger sample sizes

    #### Study Objectives:
    - **Superiority**: Prove new treatment is better than control (most common)
    - **Non-Inferiority**: Prove new treatment is not worse than active control (common when new treatment has other advantages like safety, cost, convenience)
    - **Equivalence**: Prove new treatment is therapeutically similar to reference (biosimilars, generics)

    ### When to Consult a Statistician:

    While this calculator provides rigorous calculations, you should consult a qualified statistician for:
    - **Critical studies**: Phase III trials, pivotal studies, regulatory submissions
    - **Complex designs**: Cluster randomization, crossover designs, adaptive trials, interim analyses
    - **Multiple endpoints**: Multiplicity adjustments, composite outcomes
    - **Special populations**: Rare diseases, pediatrics (ethical considerations for sample size)
    - **Unusual distributional assumptions**: Heavily skewed data, bounded outcomes
    - **Protocol development**: Ensuring all statistical aspects are properly specified
    - **Regulatory submissions**: FDA, EMA, and other agencies often require statistical review

    ### Limitations & Disclaimers:

    #### General Limitations:
    - Calculations assume all statistical test assumptions are met (normality, independence, etc.)
    - Non-parametric approximations use Asymptotic Relative Efficiency (ARE) - most accurate for large samples
    - Fisher's Exact test power is approximated using normal theory with adjustments
    - Survival analysis assumes proportional hazards (constant hazard ratio over time)

    #### This Calculator Does NOT Handle:
    - Cluster randomized trials (requires inflation for intra-cluster correlation)
    - Crossover designs (requires modeling of period effects and carryover)
    - Multiple comparison adjustments (Bonferroni, FDR, etc.)
    - Interim analysis adjustments (O'Brien-Fleming, Lan-DeMets alpha spending)
    - Complex adaptive designs
    - Bayesian sample size determination
    - Non-inferiority margins (must be specified separately)

    #### Best Practice Recommendations:
    1. **Verify assumptions**: Check that your data will meet test requirements
    2. **Sensitivity analyses**: Try different effect size scenarios
    3. **Document everything**: Record all assumptions and their sources
    4. **Pilot studies**: Use pilot data to refine effect size estimates
    5. **Pre-registration**: Register your analysis plan (e.g., clinicaltrials.gov, OSF)
    6. **Conservative approach**: When uncertain, use larger samples
    7. **Statistical collaboration**: Involve a statistician early in study planning

    ### Reproducibility:

    All calculations use established methods from:
    - **statsmodels** (Python statistical library)
    - **scipy** (Scientific Python)

    See the "Library Versions & Reproducibility" section at the bottom for exact versions.

    ### References & Further Reading:

    - **Cohen, J. (1988)**. Statistical power analysis for the behavioral sciences (2nd ed.). [Classic reference for effect sizes]
    - **ICH E9 (1998)**. Statistical principles for clinical trials. [Regulatory guideline]
    - **Julious, S. A. (2010)**. Sample sizes for clinical trials. CRC Press. [Comprehensive guide]
    - **Chow, S. C., et al. (2017)**. Sample size calculations in clinical research (3rd ed.). CRC Press. [Practical handbook]

    ### Support & Feedback:

    This calculator is provided as-is for educational and research planning purposes. Results should be validated
    and critically evaluated by qualified researchers. For questions about statistical methods or study design,
    consult with a biostatistician or statistical collaborator.

    **Version 1.3** - Enhanced with comprehensive guidance, practical context, and advanced methods (Cluster-Randomized, Repeated Measures, Bayesian).
    """)

# Sidebar
st.sidebar.title("Setup")

# Test selection guide
if st.sidebar.checkbox("Show Test Selection Guide", key="show_guide"):
    show_test_selection_guide()
else:
    show_main_interface()

# Footer
st.divider()
with st.expander("Library Versions & Reproducibility"):
    import statsmodels
    import scipy

    versions = [
        f"streamlit=={st.__version__}",
        f"numpy=={np.__version__}",
        f"scipy=={scipy.__version__}",
        f"statsmodels=={statsmodels.__version__}",
        f"pandas=={pd.__version__}"
    ]

    st.caption("Powered by:")
    for v in versions:
        st.caption(f"- {v}")

    st.caption("---")
    st.caption("For reproducibility, save as `requirements.txt`:")
    st.code("\n".join(versions), language='text')

st.caption("Timothy P. Copeland | www.tcope.land | ©2025 | Calculator Version: 1.3 (Advanced Methods) | Geneva, CH")
