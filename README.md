# Analytical-Lyapunov-Function-Discovery
Offical Pytorch implementation for the paper **"Analytical Lyapunov Function Discovery: An RL-based Generative Approach"**, presented at ICML 2025. We introduce the first RL-based framework for directly discovering analytical Lyapunov functions for nonlinear dynamical systems, bypassing the need for supervised learning with large-scale datasets. Our framework succeeds on various non-polynomial dynamics, like the simple pendulum, quadrotor, and power system frequency control, and notably scales to a 10-D system and discovers a valid local Lyapunov function for power system frequency control with lossy transmission lines, which is previously unknown in the literature. For details, see [**Analytical Lyapunov Function Discovery: An RL-based Generative Approach**](https://arxiv.org/abs/2502.02014).


## Installation
Clone this repository and install the required Conda environment and dependencies by running:

```bash
run install.sh
```

## Input Test Dynamics and Training Parameters

Before training, you need to specify the test dynamics and parameters defining the desirable output expressions. Follow these steps:

### Define the Symbolic ODEs

In [config.py](config.py), edit the function ``dynamics()`` to define your system's ODEs. Specify the state-space variables and return the dynamics in symbolic form.

### Specify the State Space for Local Stability Analysis


Modify the state space domain $\mathcal{D}$ and the size of training set $\mathcal{X}$ in [libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv](libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv). Assign the corresponding entry name (e.g., fn_d_all_x) to the variable ``conf.exp.benchmark`` in
[Lyapunov_test_dso.py](Lyapunov_test_dso.py) at Line 67.


### Set Training Parameters


In [config.py](config.py), update the ``config_factory()`` function to customize training hyperparameters, such as:

* symbolic library $\mathcal{L}$,

* length of sampled candidates (max \& min),

* learning rates of training,

* hyperparameter $\epsilon$ of risk-seeking policy gradient and so on.

**Note**: In folder example_config, you can find a few examples on how to define system dynamics and relevant training parameters.

## Training
Once your dynamics and training domain are configured, start training by running:
```bash
python Lyapunov_test_dso.py
```

Intermediate training statistics and final training result (if training converges) can be found in the log directory ``./log/{$RUN}``.

To configure and test the experiments with different initialization, please modify at the top of [Lyapunov_test_dso.py](Lyapunov_test_dso.py)
```python
conf.exp.seed_start = 13
```


## Citation
If you found this work useful or interesting for your own research, we would appreciate if you could cite our work:
```
@article{zou2025analytical,
  title={Analytical Lyapunov Function Discovery: An RL-based Generative Approach},
  author={Zou, Haohan and Feng, Jie and Zhao, Hao and Shi, Yuanyuan},
  journal={arXiv preprint arXiv:2502.02014},
  year={2025}
}
```

Feel free to leave any questions in the issues of Github or email the author at hazou@ucsd.edu.