from tests.normalization import run_normalization_test
from tests.feature_selection import run_feature_selection_test
from tests.tuning import run_hyperparameter_tuning_test
from tests.imputation import run_value_imputation_test

def run_tests(test_type):
    if test_type == "normalization":
        run_normalization_test()
    elif test_type == "feature_selection":
        run_feature_selection_test()
    elif test_type == "tuning":
        run_hyperparameter_tuning_test()
    elif test_type == "imputation":
        run_value_imputation_test()
    else:
        print("Invalid test type. Please choose from: normalization, feature_selection, tuning, or imputation.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <test_type>")
        print("Where <test_type> is one of: normalization, feature_selection, tuning, imputation")
        sys.exit(1)

    test_type = sys.argv[1]
    run_tests(test_type)