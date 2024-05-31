import numpy as np
from scipy.stats import shapiro, levene, f_oneway

def check_assumptions(scores):
    # Shapiro-Wilk test for normality
    normality_tests = [shapiro(group) for group in scores]
    normality_passed = all(p_value > 0.05 for _, p_value in normality_tests)
    
    # Levene's test for homogeneity of variances
    homogeneity_test = levene(*scores)
    homogeneity_passed = homogeneity_test.pvalue > 0.05
    
    return normality_passed, homogeneity_passed

def check_significance(scores):
    # Check assumptions
    normality_passed, homogeneity_passed = check_assumptions(scores)
    
    if not normality_passed:
        print("Normality assumption violated. ANOVA may not be appropriate.")
    elif not homogeneity_passed:
        print("Homogeneity of variances assumption violated. ANOVA may not be appropriate.")
    else:
        print("Anova test is valid to apply")

        # Perform one-way ANOVA test
        f_statistic, p_value = f_oneway(*scores)

        if p_value < 0.05:
            print("Statistically significant difference detected (p < 0.05)")
        else:
            print("No statistically significant difference detected (p >= 0.05)")

        # Output result
        print('p_value:', p_value)


original_order = [0.7063492063492064, 0.7103174603174603, 0.6845238095238095, 0.7202380952380952, 0.703042328042328]
electrode_shuffled_order = [0.628968253968254, 0.6798941798941799, 0.6633597883597884, 0.625, 0.6818783068783069]
time_shuffled_order = [0.5456349206349206, 0.5152116402116402, 0.5403439153439153, 0.5509259259259259, 0.5535714285714286]

# First Experiment Results
electores_shuffled_exp_results = [
    original_order,
    electrode_shuffled_order,
]

# Second Experiment Results
time_shuffled_exp_results = [
    original_order,
    time_shuffled_order
]

# This Experiment Results
both_shuffled_exp_results = [
    electrode_shuffled_order,
    time_shuffled_order
]

check_significance(both_shuffled_exp_results)
