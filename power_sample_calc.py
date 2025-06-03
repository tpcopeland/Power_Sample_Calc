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
    """Check expected counts for proportion tests."""
    if test_type == "two_prop":
        p1, p2 = raw_inputs.get("prop1"), raw_inputs.get("prop2")
        if n1 is not None and p1 is not None and p2 is not None:
            n2 = math.ceil(n1 * n_ratio)
            counts = [n1 * p1, n1 * (1 - p1), n2 * p2, n2 * (1 - p2)]
            min_count = min(counts)
            if min_count < 5:
                msg = f"Expected cell count < 5 ({min_count:.1f}). Consider Fisher's Exact Test."
                (st.warning if show_warning else st.sidebar.warning)(msg)
    elif test_type == "one_prop":
        null_p = raw_inputs.get("null_prop")
        if n1 is not None and null_p is not None:
            counts = [n1 * null_p, n1 * (1 - null_p)]
            min_count = min(counts)
            if min_count < 5:
                msg = f"Expected count < 5 ({min_count:.1f}). Z-test may be unreliable."
                (st.warning if show_warning else st.sidebar.warning)(msg)


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

    # Debug information
    debug_info = {
        "alpha": alpha,
        "alternative": alternative,
        "power": power,
        "nobs1": nobs1,
        "sample_prop": sample_prop,
        "null_prop": null_prop
    }

    # Input validation with detailed error messages
    if sample_prop is None or null_prop is None:
        st.error("Missing proportion values. Please provide both null proportion and sample proportion.")
        st.write("Debug info:", debug_info)
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

        if power is None and nobs1:  # Calculate power
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
        st.write("Debug info:", debug_info)
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
        # Show debug information for troubleshooting
        with st.expander("Debug Information"):
            st.write("Inputs:", inputs)
            st.write("Configuration:", config)


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

    # Alternative hypothesis
    if config.get("fixed_alt"):
        inputs["alternative"] = "two-sided"
    else:
        alt_map = {"Two-sided": "two-sided", "One-sided (larger)": "larger", "One-sided (smaller)": "smaller"}
        alt_choice = st.sidebar.selectbox("Alternative Hypothesis", list(alt_map.keys()), key=f"alt_{key}")
        inputs["alternative"] = alt_map[alt_choice]

    # Effect size (if not calculating MDES)
    if inputs["goal"] != "MDES":
        effect_inputs = collect_effect_size_inputs(config, key)
        inputs.update(effect_inputs)

    # k groups for ANOVA/KW
    if config.get("k_groups"):
        inputs["k_groups"] = st.sidebar.number_input("4. Number of Groups (k)", min_value=3, value=3, key=f"k_{key}")

    # Sample size inputs
    if config.get("n_ratio"):
        inputs["n_ratio"] = st.sidebar.number_input("5. Sample Size Ratio (N₂/N₁)", 0.1, 10.0, 1.0, 0.1,
                                                    key=f"ratio_{key}")

    if inputs["goal"] != "Sample Size":
        n_label = "Sample Size (N)" if config.get("nobs_total") else "Sample Size Group 1 (N₁)"
        inputs["n"] = st.sidebar.number_input(f"6. {n_label}", min_value=3, value=30, key=f"n_{key}")

    # Dropout
    if inputs["goal"] == "Sample Size":
        inputs["dropout"] = st.sidebar.slider("Dropout Rate (%)", 0, 50, 0, 1, key=f"dropout_{key}")

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

    # Check if raw inputs available
    raw_available = bool(config.get("raw_inputs"))

    # For single proportion test, force raw values since we need actual proportions
    if config.get("key") == "singleprop":
        method = "Raw Values"
        st.sidebar.info("ℹ️ Single proportion tests require actual proportion values")

        # Show current inputs for debugging
        if st.sidebar.checkbox("Show debug info", key=f"debug_{key}"):
            st.sidebar.write("Current session state keys:")
            debug_keys = [k for k in st.session_state.keys() if key in k]
            st.sidebar.write(debug_keys)
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

        # Debug: Show what we collected
        if config.get("key") == "singleprop" and st.sidebar.checkbox("Show collected values", key=f"debug_vals_{key}"):
            st.sidebar.write("Collected raw values:", raw_vals)

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

            if func_name == "calculate_single_proportion_power":
                raw_vals = inputs.get("raw_vals", {})
                result = calculate_single_proportion_power(
                    alpha=alpha,
                    alternative=alt,
                    power=power if goal != "Power" else None,
                    nobs1=n if goal != "Sample Size" else None,
                    sample_prop=raw_vals.get("sample_prop"),
                    null_prop=raw_vals.get("null_prop")
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

        if goal == "Power":
            result = max(0.0, min(1.0, result))

    return result


def display_results(config: Dict, inputs: Dict, result: float):
    """Display calculation results."""
    goal = inputs["goal"]

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

    elif goal == "Power":
        st.metric("Calculated Power (1-β)", f"{result:.3f}")
        st.write(f"Probability of detecting the effect: {result:.1%}")

    elif goal == "MDES":
        st.metric("MDES", f"{result:.3f}")
        st.write(f"Smallest detectable standardized effect: {result:.3f}")

    # Summary table
    st.markdown("---")
    st.subheader("Summary of Inputs")
    summary = {
        "Parameter": ["Calculation Goal", "Alpha (α)", "Alternative"],
        "Value": [goal, inputs["alpha"], inputs["alternative"]]
    }

    if "effect_size" in inputs:
        summary["Parameter"].append("Effect Size")
        summary["Value"].append(f"{inputs['effect_size']:.3f}" if inputs['effect_size'] else "N/A")

    if "n" in inputs:
        summary["Parameter"].append("Sample Size")
        summary["Value"].append(inputs["n"])

    if "power" in inputs:
        summary["Parameter"].append("Power (1-β)")
        summary["Value"].append(inputs["power"])

    display_results_table(summary)


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

    else:
        st.warning("Survival analysis not supported.")
        return

    st.success(f"**Recommended: {test}**")

    if st.button(f"Use {test}", type="primary"):
        # Set the session state values using a different key to avoid conflict
        st.session_state["selected_test_category"] = category
        st.session_state["selected_test_name"] = test
        st.session_state["guide_selection_made"] = True
        # Hide the guide after a selection is made so we jump to the main
        # calculator with the recommended test already chosen
        st.session_state["show_guide"] = False
        st.rerun()


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
    categories = ["Parametric Tests", "Non-Parametric Tests"]
    category = st.sidebar.radio("**1. Select Test Category:**", categories,
                                key="test_category",
                                help="Parametric: Assume specific distribution. Non-Parametric: Fewer assumptions.")

    # Test selection
    if category == "Parametric Tests":
        tests = ["Two-Sample Independent Groups t-test", "One-Sample t-test", "Paired t-test",
                 "Z-test: Two Independent Proportions", "Z-test: Single Proportion",
                 "One-Way ANOVA (Between Subjects)"]
    else:
        tests = ["Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Kruskal-Wallis Test",
                 "Fisher's Exact Test"]

    selected_test = st.sidebar.radio("Select Specific Test:", tests, key="selected_test")

    st.sidebar.divider()

    # Run calculation
    if selected_test:
        run_test_calculation(selected_test)
    else:
        st.info("👈 Please select a statistical test from the sidebar to begin.")


# ==============================================================================
#                             MAIN APP
# ==============================================================================
st.set_page_config(page_title="Power Calculator", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 Power and Sample Size Calculator")

# About section
with st.expander("About this Calculator", expanded=False):
    st.markdown("""
    ### Overview

    This tool assists scientists, particularly in pharmaceutical and medical device research, in determining appropriate sample sizes or estimating statistical power for various study designs. It aims to be accessible even with minimal prior statistical experience.

    #### Key Features:
    - Calculate **Sample Size (N)**, **Statistical Power (1-β)**, or **Minimum Detectable Effect Size (MDES)**.
    - Interactive **Test Selection Guide** to help choose the appropriate statistical test based on outcome type and study design.
    - Support for common **Parametric Tests**:
        - Independent Samples t-test
        - One-Sample t-test
        - Paired t-test
        - Z-test for Two Independent Proportions
        - Z-test for Single Proportion
        - One-Way ANOVA (Between Subjects)
    - Support for common **Non-Parametric Tests** (using ARE-based or heuristic approximations):
        - Mann-Whitney U Test (Wilcoxon Rank-Sum)
        - Wilcoxon Signed-Rank Test
        - Kruskal-Wallis Test
        - Fisher's Exact Test
    - Input effect size using **standardized metrics** (Cohen's d, f, h) or **raw values** (means, medians, proportions + variability estimates).
    - Tooltips and expandable sections provide **explanations** of statistical concepts, assumptions, and parameters.
    - Option to adjust sample size calculations for anticipated **dropout rates**.
    - **Summary tables** detail the inputs used for each calculation, aiding reproducibility.

    #### How to Use:
    1.  **(Optional)** Check the "Show Test Selection Guide" box in the sidebar for interactive help choosing the right test.
    2.  Select the **Test Category** (Parametric/Non-Parametric) and the specific **Statistical Test** from the sidebar menus.
    3.  Choose what you want to **Calculate** (Sample Size, Power, or MDES).
    4.  Enter the required parameters in the sidebar that appears for the selected test. Use the `(?)` icons next to inputs for detailed explanations.
    5.  View the calculated **Results** and the **Summary of Inputs Used** in the main panel.
    6.  Use the **"Reset Inputs for This Test"** button in the sidebar to clear parameters for the current module and start fresh.

    #### Disclaimer:
    This tool provides calculations based on established statistical formulas and approximations implemented in Python libraries (`statsmodels`, `scipy`). Approximations for non-parametric tests and Fisher's Exact test power have limitations, especially with very small samples. Results should be critically evaluated in the context of your specific research goals and assumptions. **Consultation with a qualified statistician is strongly recommended for designing critical studies or interpreting results.**
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

st.caption("Timothy P. Copeland | www.tcope.land | ©2025 | Calculator Version: 1.0 | Geneva, CH")
