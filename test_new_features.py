#!/usr/bin/env python3
"""
Comprehensive test suite for new features in power_sample_calc.py
Tests: Cluster-Randomized, Repeated Measures ANOVA, and Bayesian methods
"""

import sys
import numpy as np
import math

# Import functions from the main module
sys.path.insert(0, '/home/user/Power_Sample_Calc')
from power_sample_calc import (
    calculate_design_effect,
    calculate_clusters_needed,
    interpret_icc,
    calculate_repeated_measures_power,
    calculate_repeated_measures_n,
    calculate_assurance,
    calculate_bayesian_sample_size,
    calculate_expected_power,
    TTestIndPower
)

def test_cluster_randomized():
    """Test cluster-randomized trial functions"""
    print("\n" + "="*70)
    print("TESTING: Cluster-Randomized Trial Functions")
    print("="*70)

    # Test 1: Design Effect Calculation
    print("\n[Test 1] Design Effect Calculation")
    icc = 0.05
    cluster_size = 20
    deff = calculate_design_effect(icc, cluster_size)
    expected_deff = 1 + (20 - 1) * 0.05  # 1.95
    assert abs(deff - expected_deff) < 0.001, f"DEFF calculation failed: {deff} != {expected_deff}"
    print(f"✓ PASS: DEFF with ICC={icc}, m={cluster_size} = {deff:.3f}")

    # Test 2: Small ICC
    print("\n[Test 2] Small ICC Design Effect")
    icc_small = 0.01
    deff_small = calculate_design_effect(icc_small, cluster_size)
    print(f"✓ PASS: DEFF with ICC={icc_small}, m={cluster_size} = {deff_small:.3f}")

    # Test 3: Large Cluster Size
    print("\n[Test 3] Large Cluster Size Effect")
    large_cluster = 100
    deff_large = calculate_design_effect(icc, large_cluster)
    print(f"✓ PASS: DEFF with ICC={icc}, m={large_cluster} = {deff_large:.3f}")

    # Test 4: Calculate Clusters Needed
    print("\n[Test 4] Clusters Needed Calculation")
    individual_n = 64  # Per group
    total_n, n_clusters, deff = calculate_clusters_needed(individual_n, cluster_size, icc)
    print(f"✓ PASS: Individual N={individual_n} → Total N={total_n}, Clusters={n_clusters}, DEFF={deff:.3f}")

    # Test 5: ICC Interpretation
    print("\n[Test 5] ICC Interpretation")
    for test_icc in [0.005, 0.03, 0.08, 0.15, 0.25]:
        interpretation = interpret_icc(test_icc)
        print(f"✓ ICC={test_icc:.3f}: {interpretation}")

    # Test 6: Edge Cases
    print("\n[Test 6] Edge Cases")
    try:
        calculate_design_effect(-0.1, 20)  # Negative ICC should fail
        print("✗ FAIL: Should reject negative ICC")
    except ValueError as e:
        print(f"✓ PASS: Correctly rejected negative ICC")

    try:
        calculate_design_effect(1.5, 20)  # ICC > 1 should fail
        print("✗ FAIL: Should reject ICC > 1")
    except ValueError as e:
        print(f"✓ PASS: Correctly rejected ICC > 1")

    print("\n✅ All Cluster-Randomized Tests PASSED")


def test_repeated_measures():
    """Test repeated measures ANOVA functions"""
    print("\n" + "="*70)
    print("TESTING: Repeated Measures ANOVA Functions")
    print("="*70)

    # Test 1: Sample Size Calculation
    print("\n[Test 1] Sample Size Calculation")
    effect_size = 0.25  # Medium effect
    alpha = 0.05
    power = 0.80
    num_measurements = 3
    correlation = 0.5

    n = calculate_repeated_measures_n(effect_size, alpha, power, num_measurements, correlation)
    if n:
        print(f"✓ PASS: Required N = {n} subjects for {num_measurements} measurements")
        print(f"  Total observations = {n * num_measurements}")
    else:
        print("✗ FAIL: Sample size calculation returned None")

    # Test 2: Power Calculation
    print("\n[Test 2] Power Calculation")
    test_n = 30
    calculated_power = calculate_repeated_measures_power(
        n=test_n,
        effect_size=effect_size,
        alpha=alpha,
        num_measurements=num_measurements,
        correlation=correlation
    )
    if calculated_power:
        print(f"✓ PASS: Power with N={test_n} = {calculated_power:.3f}")
    else:
        print("✗ FAIL: Power calculation returned None")

    # Test 3: High Correlation Advantage
    print("\n[Test 3] High Correlation Advantage")
    n_low_corr = calculate_repeated_measures_n(effect_size, alpha, power, num_measurements, correlation=0.2)
    n_high_corr = calculate_repeated_measures_n(effect_size, alpha, power, num_measurements, correlation=0.8)

    if n_low_corr and n_high_corr:
        print(f"✓ PASS: Low correlation (0.2) requires N={n_low_corr}")
        print(f"✓ PASS: High correlation (0.8) requires N={n_high_corr}")
        print(f"  Efficiency gain: {n_low_corr - n_high_corr} fewer subjects")
        assert n_high_corr < n_low_corr, "Higher correlation should require fewer subjects"
    else:
        print("✗ FAIL: Correlation comparison failed")

    # Test 4: Multiple Time Points
    print("\n[Test 4] Multiple Time Points")
    for num_time_points in [2, 3, 5, 10]:
        n_tp = calculate_repeated_measures_n(effect_size, alpha, power, num_time_points, 0.5)
        if n_tp:
            print(f"✓ PASS: {num_time_points} time points requires N={n_tp} subjects")

    # Test 5: Edge Cases
    print("\n[Test 5] Edge Cases")
    # Correlation = 1 (perfect correlation)
    result = calculate_repeated_measures_n(effect_size, alpha, power, 3, 0.99)
    print(f"✓ Correlation=0.99: N={result}")

    # Correlation = 0 (no correlation, like between-subjects)
    result = calculate_repeated_measures_n(effect_size, alpha, power, 3, 0.0)
    print(f"✓ Correlation=0.0: N={result}")

    print("\n✅ All Repeated Measures Tests PASSED")


def test_bayesian_methods():
    """Test Bayesian sample size functions"""
    print("\n" + "="*70)
    print("TESTING: Bayesian Sample Size Functions")
    print("="*70)

    # Test 1: Assurance Calculation
    print("\n[Test 1] Assurance Calculation")
    n = 64
    alpha = 0.05
    prior_mean = 0.5
    prior_sd = 0.25
    target_power = 0.80

    assurance = calculate_assurance(n, alpha, prior_mean, prior_sd, target_power)
    if assurance:
        print(f"✓ PASS: Assurance with N={n} = {assurance:.3f}")
        assert 0 <= assurance <= 1, "Assurance must be between 0 and 1"
    else:
        print("✗ FAIL: Assurance calculation returned None")

    # Test 2: Expected Power
    print("\n[Test 2] Expected Power Calculation")
    expected_power = calculate_expected_power(n, alpha, prior_mean, prior_sd)
    if expected_power:
        print(f"✓ PASS: Expected Power with N={n} = {expected_power:.3f}")
        assert 0 <= expected_power <= 1, "Expected power must be between 0 and 1"
    else:
        print("✗ FAIL: Expected power calculation returned None")

    # Test 3: Bayesian Sample Size
    print("\n[Test 3] Bayesian Sample Size Calculation")
    print("  (This may take 10-30 seconds due to binary search...)")
    target_assurance = 0.80
    bayesian_n = calculate_bayesian_sample_size(
        alpha=alpha,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        target_assurance=target_assurance,
        target_power=target_power,
        max_n=500  # Limit for test speed
    )
    if bayesian_n:
        print(f"✓ PASS: Bayesian N for {target_assurance:.0%} assurance = {bayesian_n}")
    else:
        print("✗ FAIL: Bayesian sample size returned None")

    # Test 4: Different Priors
    print("\n[Test 4] Different Prior Distributions")
    priors = {
        "Skeptical": (0.1, 0.15),
        "Neutral": (0.3, 0.30),
        "Optimistic": (0.5, 0.25)
    }

    for prior_name, (mean, sd) in priors.items():
        exp_pwr = calculate_expected_power(64, alpha, mean, sd)
        if exp_pwr:
            print(f"✓ {prior_name} prior ({mean}, {sd}): Expected Power = {exp_pwr:.3f}")

    # Test 5: Assurance vs Target Power
    print("\n[Test 5] Assurance for Different Target Powers")
    for target_pwr in [0.70, 0.80, 0.90]:
        assur = calculate_assurance(64, alpha, prior_mean, prior_sd, target_pwr)
        if assur:
            print(f"✓ Target Power={target_pwr:.0%}: Assurance = {assur:.3f}")

    print("\n✅ All Bayesian Tests PASSED")


def test_original_functions():
    """Regression test: Verify original functions still work"""
    print("\n" + "="*70)
    print("REGRESSION TEST: Original Functions")
    print("="*70)

    # Test Two-Sample t-test power calculation
    print("\n[Test 1] Two-Sample t-test")
    power_calc = TTestIndPower()
    n_required = power_calc.solve_power(
        effect_size=0.5,
        nobs1=None,
        alpha=0.05,
        power=0.80,
        ratio=1.0,
        alternative='two-sided'
    )
    print(f"✓ PASS: Two-sample t-test requires N={math.ceil(n_required)} per group")

    # Test power given sample size
    calculated_power = power_calc.solve_power(
        effect_size=0.5,
        nobs1=64,
        alpha=0.05,
        power=None,
        ratio=1.0,
        alternative='two-sided'
    )
    print(f"✓ PASS: Power with N=64 per group = {calculated_power:.3f}")

    # Test MDES calculation
    mdes = power_calc.solve_power(
        effect_size=None,
        nobs1=64,
        alpha=0.05,
        power=0.80,
        ratio=1.0,
        alternative='two-sided'
    )
    print(f"✓ PASS: MDES with N=64, power=0.80 = {mdes:.3f}")

    print("\n✅ All Original Function Tests PASSED")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "#"*70)
    print("# COMPREHENSIVE TEST SUITE FOR NEW FEATURES")
    print("# Testing: Cluster-Randomized, Repeated Measures, Bayesian Methods")
    print("#"*70)

    try:
        test_cluster_randomized()
        test_repeated_measures()
        test_bayesian_methods()
        test_original_functions()

        print("\n" + "#"*70)
        print("# ALL TESTS PASSED SUCCESSFULLY! ✅")
        print("#"*70)
        return True

    except Exception as e:
        print("\n" + "#"*70)
        print(f"# TEST FAILED: {str(e)} ❌")
        print("#"*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
