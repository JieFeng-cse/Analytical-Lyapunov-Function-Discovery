import numpy as np
import pandas as pd

from dso.task import HierarchicalTask
from dso.library import Library
from dso.functions import create_tokens
from dso.task.regression.dataset import BenchmarkDataset
import sympy as sym
from sympy import lambdify
import timeit
import sympytorch
from config import dynamics



class RegressionTask(HierarchicalTask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "regression"

    def __init__(self, function_set, dataset, metric="lyapunov_risk",
                 metric_params=(1.0,), extra_metric_test=None,
                 extra_metric_test_params=(), reward_noise=0.0,
                 reward_noise_type="r", threshold=1e-12,
                 normalize_variance=False, protected=False,
                 decision_tree_threshold_set=None, n_input_var=None):
        """
        Parameters
        ----------
        function_set : list or None
            List of allowable functions. If None, uses function_set according to
            benchmark dataset.

        dataset : dict, str, or tuple
            If dict: .dataset.BenchmarkDataset kwargs.
            If str ending with .csv: filename of dataset.
            If other str: name of benchmark dataset.
            If tuple: (X, y) data

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        extra_metric_test : str
            Name of extra function metric to use for testing.

        extra_metric_test_params : list
            List of metric-specific parameters for extra test metric.

        reward_noise : float
            Noise level to use when computing reward.

        reward_noise_type : "y_hat" or "r"
            "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
            "r" : N(0, reward_noise) is added to r.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision trees.
        """

        super(HierarchicalTask).__init__()
        

        """
        Configure (X, y) train/test data. There are four supported use cases:
        (1) named benchmark, (2) benchmark config, (3) filename, and (4) direct
        (X, y) data.
        """
        self.X_test = self.y_test = self.y_test_noiseless = None
        self.mse_weight = 0
        
        # Case 1: Named benchmark dataset (shortcut for Case 2)
        if isinstance(dataset, str) and not dataset.endswith(".csv"):
            
            dataset = {"name": dataset}


        # Case 2: Benchmark dataset config
        if isinstance(dataset, dict):
            benchmark = BenchmarkDataset(n_input_var=n_input_var, **dataset)
            self.X_train = benchmark.X_train
            self.y_train = benchmark.y_train
            self.X_test = benchmark.X_test
            self.X_test_ood = benchmark.X_test_ood
            self.y_test_ood = benchmark.y_test_ood
            self.y_test = benchmark.y_test
            self.y_test_noiseless = benchmark.y_test_noiseless
            self.name = benchmark.name
            self.function_set_name = benchmark.function_set_name
            self.train_spec = benchmark.train_spec
            self.test_spec = benchmark.test_spec
            self.dataset_size_multiplier = benchmark.dataset_size_multiplier
            self.n_input_var = benchmark.n_input_var
            self.numpy_expr = benchmark.numpy_expr
            output_message = 'Function set                   : {} --> {}\n'.format(
                self.function_set_name, function_set)
            print(output_message)

            # For benchmarks, always use the benchmark function_set.
            # Issue a warning if the user tried to supply a different one.
            # if function_set is not None and function_set != benchmark.function_set:
            #     print("WARNING: function_set provided when running benchmark "
            #           "problem. The provided function_set will be ignored; the "
            #           "benchmark function_set will be used instead.\nProvided "
            #           "function_set:\n  {}\nBenchmark function_set:\n  {}."
            #           .format(function_set, benchmark.function_set))
            # function_set = benchmark.function_set

        # Case 3: Dataset filename
        elif isinstance(dataset, str) and dataset.endswith("csv"):
            # Assuming data file does not have header rows
            df = pd.read_csv(dataset, header=None)
            self.X_train = df.values[:, :-1]
            self.y_train = df.values[:, -1]
            self.name = dataset.replace("/", "_")[:-4]

        # Case 4: sklearn-like (X, y) data
        elif isinstance(dataset, tuple):
            self.X_train = dataset[0]
            self.y_train = dataset[1]
            self.name = "regression"
        
        # If not specified, set test data equal to the training data
        if self.X_test is None:
            self.X_test = self.X_train
            self.y_test = self.y_train
            self.y_test_noiseless = self.y_test

        # Save time by only computing data variances once
        self.var_y_test = np.var(self.y_test)
        self.var_y_test_noiseless = np.var(self.y_test_noiseless)

        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        self.metric, self.invalid_reward, self.max_reward = make_regression_metric(
            metric, self.y_train, *metric_params)
        self.extra_metric_test = extra_metric_test
        if extra_metric_test is not None:
            self.metric_test, _, _ = make_regression_metric(
                extra_metric_test, self.y_test, *extra_metric_test_params)
        else:
            self.metric_test = None

        """
        Configure reward noise.
        """
        self.reward_noise = reward_noise
        self.reward_noise_type = reward_noise_type
        self.normalize_variance = normalize_variance
        assert reward_noise >= 0.0, "Reward noise must be non-negative."
        if reward_noise > 0:
            assert reward_noise_type in [
                "y_hat", "r"], "Reward noise type not recognized."
            self.rng = np.random.RandomState(0)
            y_rms_train = np.sqrt(np.mean(self.y_train ** 2))
            if reward_noise_type == "y_hat":
                self.scale = reward_noise * y_rms_train
            elif reward_noise_type == "r":
                self.scale = reward_noise
        else:
            self.rng = None
            self.scale = None
        n_input_var = self.X_train.shape[1]
        # Set the Library
        tokens = create_tokens(n_input_var=n_input_var,
                               function_set=function_set,
                               protected=protected,
                               decision_tree_threshold_set=decision_tree_threshold_set)
        self.library = Library(tokens)

        # Set stochastic flag
        self.stochastic = reward_noise > 0.0

    def reward_function(self, p):

        # Compute estimated values
        y_hat = p.execute(self.X_train)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            return self.invalid_reward

        # Observation noise
        # For reward_noise_type == "y_hat", success must always be checked to
        # ensure success cases aren't overlooked due to noise. If successful,
        # return max_reward.
        if self.reward_noise and self.reward_noise_type == "y_hat":
            if p.evaluate.get("success"):
                return self.max_reward
            y_hat += self.rng.normal(loc=0, scale=self.scale, size=y_hat.shape)

        # Compute metric
        evaluated_origin = {str(i): 0 for i in dynamics()[0]}
        deri = p.sympy_expr[0]
        origin = float(deri.evalf(subs=evaluated_origin))
    
        variable_count = len(list(p.sympy_expr[0].free_symbols))
        if variable_count < len(dynamics()[0]):
            return self.invalid_reward
        else:
            lie = p.sympy_expr_lie
            if len(list(lie.free_symbols)) == 0:
                if lie.is_positive:
                    lie_result = np.ones(self.X_train.shape[0]) * 10
                else:
                    lie_result = np.ones(self.X_train.shape[0]) * (-0.5)
            
            numpy_v_dot = lambdify(dynamics()[0], lie, "numpy")
            lie_result = numpy_v_dot(*[self.X_train[:, i] for i in range(self.X_train.shape[1])]) * 1

        r = self.metric(lie_result, y_hat-origin)

        # Direct reward noise
        # For reward_noise_type == "r", success can for ~max_reward metrics be
        # confirmed before adding noise. If successful, must return np.inf to
        # avoid overlooking success cases.
        if self.reward_noise and self.reward_noise_type == "r":
            if r >= self.max_reward - 1e-5 and p.evaluate.get("success"):
                return np.inf
            r += self.rng.normal(loc=0, scale=self.scale)
            if self.normalize_variance:
                r /= np.sqrt(1 + 12 * self.scale ** 2)

        return r
    
    
    def evaluate(self, p):

        def relu(x):
            return x * (x > 0)

        # Compute predictions on test data
        y_hat = p.execute(self.X_test)


        if p.invalid:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            evaluated_origin = {str(i): 0 for i in dynamics()[0]}

            deri = p.sympy_expr[0]
            origin = float(deri.evalf(subs=evaluated_origin))

            if len(list(p.sympy_expr[0].free_symbols)) < len(dynamics()[0]):
                lie_result = np.ones(self.X_test.shape[0]) * 10
            else:
                lie = p.sympy_expr_lie
                if len(list(lie.free_symbols)) == 0:
                    if lie.is_positive:
                        lie_result = np.ones(self.X_test.shape[0]) * 10
                    else:
                        lie_result = np.zeros(self.X_test.shape[0])
                numpy_v_dot = lambdify(dynamics()[0], lie, "numpy")
                lie_result = numpy_v_dot(*[self.X_test[:, i] for i in range(self.X_test.shape[1])])


            # NMSE on test data (used to report final error)
            y_hat = y_hat
            nmse_test = np.mean(relu(-(y_hat - origin))) + np.mean(relu(lie_result - 0.000))

            # NMSE on noiseless test data (used to determine recovery)
            nmse_test_noiseless = nmse_test

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test_noiseless < self.threshold

        info = {
            "nmse_test": nmse_test,
            "nmse_test_noiseless": nmse_test_noiseless,
            "success": success
        }


        return info


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """


    def relu(x):
        return x * (x > 0)

    all_metrics = {

        # Lyapunov risk
        # Range: [0, 1]
        "lyapunov_risk":(lambda y, y_hat: 1 / (1.0+ (2.5 * np.mean(relu(-(y_hat-0.05))) + np.mean(relu(y+0.2))) * (1)),
                    1),
    
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(
        name, all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    all_invalid_rewards = {
        "lyapunov_risk": 0.0,
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "lyapunov_risk": 1.0,
    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward
