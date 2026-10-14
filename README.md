# Black-Box Optimisation Capstone Project

## 1. Project Overview

This project explores **black-box optimisation (BBO)**, where the objective function is unknown and must be optimised through queries. Each query provides an input configuration and returns an observed objective value, allowing a model to learn about the function without knowing its internal structure.

The goal is to identify high-performing input configurations while minimising the number of evaluations required. This reflects real-world machine learning problems where evaluations can be expensive, including hyperparameter tuning, experimental design, engineering optimisation, and scientific experimentation.

Through this project, I aim to develop practical skills in surrogate modelling, and data-driven optimisation.

## 2. Inputs and Outputs

The optimisation functions accept numerical input vectors whose values are constrained to the interval [0, 1]. Each vector represents a candidate configuration, with the number of dimensions depending on the function being evaluated.

For example, a query containing four dimensions could be formatted as:

```text
0.125000-0.450000-0.780000-0.230000
```

The query is submitted to the black-box evaluator, which returns an objective value representing the quality of that configuration. The value is used to update the dataset and guide subsequent query selection.

The project loads initial observations from NumPy files and incorporates previous evaluations from CSV history files. Each observation consists of an input vector and its corresponding objective value.

## 3. Challenge Objectives

The primary objective is to find the best possible solution using a limited query budget. The challenge includes multiple unknown objective functions, and their mathematical structures are not assumed to be available.

The implementation supports both minimisation and maximisation through a configuration setting. The current default is minimisation, meaning lower objective values are preferred.

Key constraints include:

- A limited number of queries to evaluate the unknown functions.
- Unknown relationships between input dimensions and objective values.
- Potentially high-dimensional search spaces.
- The need to balance exploration of unfamiliar regions with exploitation of promising areas.

The central challenge is to make each query informative and improve the best observed result as efficiently as possible.

## 4. Technical Approach

My approach has evolved across three query submissions so far, progressing towards a more informed and structured query-selection strategy.

### Stage 1: Establishing a baseline

The initial stage focused on understanding the available data, the query format, and the optimisation objective. The priority was to establish a working process for evaluating candidate inputs and identifying the best observed result.

### Stage 2: Introducing surrogate modelling

The second stage developed a more model-driven strategy using Gaussian Process (GP) regression and expected improvement. The GP estimates the unknown objective function and its predictive uncertainty.

Expected improvement helps balance exploration and exploitation by considering both the potential to improve on the current best result and uncertainty about a candidate. Linear regression and logistic regression were also incorporated to provide additional information about trends and promising regions.

### Stage 3: Incorporating Support Vector Machines

In the third stage I replaced the logistic regression with a Support Vector Machine classifier using an RBF kernel. The classifier labels the best-performing 30% of observations as promising and learns to distinguish these from the remaining observations.

The current acquisition function combines three signals:

- **65% GP expected improvement:** prioritises candidates with potential objective improvement.
- **25% SVM score:** favours candidates classified as belonging to promising regions.
- **10% linear regression score:** provides a secondary estimate of objective trends.

The scores are normalised before being combined. Historical observations are reused when fitting the models, and previously evaluated or near-duplicate candidates are filtered out.

This approach aims to use the GP for uncertainty-aware optimisation while allowing the SVM to identify nonlinear promising regions. The weights and classification threshold remain experimental choices rather than guaranteed optimal settings.

## 5. Evaluation and Future Development

The main success criteria are the quality of the best solution found and the number of queries required to achieve it.

Future development will focus on comparing the SVM-assisted strategy against the earlier GP-based approach, testing alternative acquisition weights and promising-class thresholds, and assessing performance across different functions.

The overall aim is to build a reproducible, interpretable, and computationally practical optimisation strategy that makes better decisions from limited observations.