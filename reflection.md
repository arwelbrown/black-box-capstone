# Reflection on My Fourth BBO Iteration

### 1. Support vectors and decision boundaries

My SVM classifier labels the best-performing observations as promising and uses them to identify regions worth exploring. Points close to the decision boundary are particularly interesting because small changes in their inputs may change their predicted class. These points could reveal transitions between good and poor objective values. In my next query, I would consider candidates near this boundary, while retaining random exploration to avoid focusing too narrowly on uncertain regions.

### 2. Neural-network gradients

I extended my approach with a multilayer perceptron (MLP) surrogate to capture nonlinear relationships between inputs and objective values. Using backpropagation, I can estimate how the predicted objective changes with each input variable. For minimisation, moving in the negative gradient direction may identify inputs that reduce the predicted function value. However, these gradients describe the surrogate rather than the true objective, so I combine gradient-guided candidates with random search and Gaussian Process Expected Improvement.

### 3. Classification as good versus bad

Framing optimisation as classification provides a complementary way to identify promising regions. Logistic regression offers an interpretable, approximately linear decision boundary, while an SVM can represent more complex boundaries using a nonlinear kernel. A neural network can learn flexible nonlinear boundaries but generally requires more data and careful tuning. Misclassifying a genuinely good point may cause a valuable region to be overlooked, whereas classifying a poor point as promising wastes an evaluation. I would therefore retain exploration rather than relying entirely on predicted classes.

### 4. Choosing between models

Each model serves a different purpose. Linear regression is simple and interpretable, but may miss complex interactions. The SVM identifies promising regions based on classification, while the MLP can approximate nonlinear objective patterns. My hybrid approach balances interpretability and flexibility by combining their predictions with uncertainty-aware GP-EI. This avoids assuming that one model will perform best across every function and dimensionality.

### 5. Identifying influential input variables

The gradient of the MLP prediction indicates the local sensitivity of the objective to each input. I can compare the absolute gradient components to identify variables with the greatest local influence. Variables with larger gradients may deserve more targeted experiments, while variables with small gradients may have less influence in the current region. However, sensitivity can vary across the search space, and correlated inputs can complicate interpretation. I would assess gradients at several promising candidates before prioritising experiments.

### 6. Neural networks and classification boundaries

Backpropagation helps explain how changing an input affects the network's output, but it does not directly visualise a classification boundary. To investigate that boundary, I could train a classifier using promising and non-promising observations and plot predicted class probabilities across selected pairs of input dimensions. I would then examine whether the boundary separates observed outcomes effectively and whether uncertain regions contain potentially valuable candidates.

### 7. Nonlinearity versus complexity

Compared with linear regression and logistic regression, the MLP offers greater capacity to learn nonlinear patterns and interactions. This flexibility is valuable when the objective is complex, but it also introduces risks of overfitting, unstable predictions and additional computational cost. I have therefore retained simpler models and GP-EI as complementary signals. Ultimately, the added complexity is justified only if controlled experiments demonstrate better optimisation performance under the same evaluation budget.

### Conclusion

My fourth iteration has encouraged me to treat query selection as a balance between learning the objective, exploiting promising regions and exploring uncertainty. My next step is to compare the hybrid strategy against simpler baselines across repeated runs, using actual objective evaluations to determine whether neural-network predictions and gradient-guided search improve convergence.