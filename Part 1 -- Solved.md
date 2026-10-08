
# Homework 1


#### EE-556 Mathematics of Data - Fall 2026


In this homework, we consider a multiclass classification task modeled by multinomial (softmax) logistic regression. Your goal will be to analyze the estimator and its properties (convexity, existence/uniqueness), and to derive gradients/Hessians and smoothness bounds. The first part consists of theoretical questions only.

  

## 1. Multiclass Softmax Logistic Regression - 30 Points

  

  

We now model multiclass classification with classes $c \in \{1,\dots,C\}$. For each sample $(\mathbf{a}_i, b_i)$ with $\mathbf{a}_i \in \mathbb{R}^p$ and $b_i \in \{1,\dots,C\}$, let $\mathbf{X} = [\mathbf{x}_1,\dots,\mathbf{x}_C] \in \mathbb{R}^{p\times C}$ be the class weight matrix. The softmax model defines:

$$\mathbb{P}(b_i = c \mid \mathbf{a}_i) = \frac{\exp(\mathbf{a}_i^\top \mathbf{x}_c)}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)}$$

  

Assume i.i.d. samples $\{(\mathbf{a}_i,b_i)\}_{i=1}^n$. Our goal is to estimate $\mathbf{X}$ by maximum likelihood (and later with an $\ell_2$ regularizer).

  

### (a) (2 points)

Show that the negative log-likelihood $f$ can be written as:

  

$$\begin{aligned} f(\mathbf{X}) &= - \log \mathbb{P}(b_1,\dots,b_n\mid \mathbf{a}_1,\dots,\mathbf{a}_n)\\ &= \sum_{i=1}^n \left[ -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right]. \end{aligned}$$

  

**Solution:** We will generally denote the column of the class weight matrix corresponding to the outcome $b_i$ as $\mathbf{x}_{b_i}$. First, we derive the likelihood:

  

$$L(\mathbf{X}) = \prod_{i=1}^{n} \mathbb{P}(b_i \mid \mathbf{a}_i) = \prod_{i=1}^{n} \frac{\exp(\mathbf{a}_i^\top \mathbf{x}_{b_i})}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)}$$

  

Taking the log we get:

  

$$\log L(\mathbf{X}) = \log \left\{ \prod_{i=1}^{n} \frac{\exp(\mathbf{a}_i^\top \mathbf{x}_{b_i})}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)} \right\} = \sum_{i=1}^{n} \left[ \mathbf{a}_i^\top \mathbf{x}_{b_i} - \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right]$$

  

Negating the expression gives the desired result.

  

### (b) (3 points)

Show that $\mathbf{u} \mapsto \log\!\left(\sum_{k=1}^C e^{u_k}\right)$ is convex on $\mathbb{R}^C$. Then, show that $f(\mathbf{X})$ is convex. _Hint: use Jensen's inequality._

  

**Solution:** We wish to show that $g:\mathbf{u} \mapsto \log \sum_{k=1}^C e^{u_k}$ is convex on $\mathbb{R}^C$ (where $g:\mathbb{R}^C \rightarrow \mathbb{R}$). For $g$ to be convex on $\mathbb{R}^C$, we need:

  

$$g(\alpha \mathbf{u}_1 + (1-\alpha)\mathbf{u}_2) \leq \alpha g(\mathbf{u}_1) + (1-\alpha)g(\mathbf{u}_2)$$

  

for all $\mathbf{u}_1, \mathbf{u}_2 \in \mathbb{R}^C$ and for all $\alpha \in [0,1]$.

  

$$\log \sum_{k=1}^C \exp(\alpha u_{1,k} + (1-\alpha)u_{2,k}) \leq \alpha \log \sum_{k=1}^C \exp(u_{1,k}) + (1-\alpha) \log \sum_{k=1}^C \exp(u_{2,k})$$

  

Exponentiating both sides:

  

$$\sum_{k=1}^C \exp(\alpha u_{1,k} + (1-\alpha)u_{2,k}) \leq \exp \left\{ \log \left[ \sum_{k=1}^C \exp(u_{1,k}) \right]^\alpha + \log \left[ \sum_{k=1}^C \exp(u_{2,k}) \right]^{1-\alpha} \right\}$$

  

$$\sum_{k=1}^C \exp(\alpha u_{1,k}) \exp((1-\alpha)u_{2,k}) \leq \left[ \sum_{k=1}^C \exp(u_{1,k}) \right]^\alpha \left[ \sum_{k=1}^C \exp(u_{2,k}) \right]^{1-\alpha}$$

  

Dividing both sides:

  

$$\sum_{k=1}^C \frac{\exp(\alpha u_{1,k})}{\left[ \sum \exp(u_{1,k}) \right]^\alpha} \cdot \frac{\exp((1-\alpha)u_{2,k})}{\left[ \sum \exp(u_{2,k}) \right]^{1-\alpha}} \leq 1$$

  

Recognising the Softmax pmf, this becomes:

  

$$\sum_{k=1}^C \mathbb{P}(b_i=k \mid \mathbf{u}_1)^\alpha \mathbb{P}(b_i=k \mid \mathbf{u}_2)^{1-\alpha} \leq 1$$

  

$$\sum_{k=1}^C \mathbb{P}(b_i=k \mid \mathbf{u}_1) \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)^{1-\alpha}}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)^{1-\alpha}} \leq 1$$

  

Rewriting as an expectation:

  

$$E_{b_i \mid \mathbf{u}_1} \left\{ \left[ \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \right]^{1-\alpha} \right\} \leq 1$$

  

By Jensen's inequality:

  

$$E_{b_i \mid \mathbf{u}_1} \left\{ \left[ \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \right]^{1-\alpha} \right\} \leq \left\{ E_{b_i \mid \mathbf{u}_1} \left[ \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \right] \right\}^{1-\alpha}$$

  

This is because $y=x^\theta$ is concave by the definition of a convex function using the positive semi-definiteness of the Hessian (which here reduces to a scalar):

  

$$\frac{\partial}{\partial x}y = \theta x^{\theta-1}, \quad \frac{\partial^2}{\partial x^2}y = \theta(\theta-1)x^{\theta-2} < 0$$

  

for any $\theta \in [0,1]$ (with $\theta = 1 - \alpha$). The RHS of the Jensen inequality is:

  

$$\left\{ E_{b_i \mid \mathbf{u}_1} \left[ \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \right] \right\}^{1-\alpha} = \left( \sum_{k=1}^C \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \mathbb{P}(b_i=k \mid \mathbf{u}_1) \right)^{1-\alpha}$$

  

$$= \left( \sum_{k=1}^C \mathbb{P}(b_i=k \mid \mathbf{u}_2) \right)^{1-\alpha} = 1$$

  

since it is a pmf. This implies that:

  

$$E_{b_i \mid \mathbf{u}_1} \left\{ \left[ \frac{\mathbb{P}(b_i=k \mid \mathbf{u}_2)}{\mathbb{P}(b_i=k \mid \mathbf{u}_1)} \right]^{1-\alpha} \right\} \leq 1$$

  

### (c) (1 point)

Since the negative log-likelihood $f$ is convex, every local minimizer is a global minimizer. The maximum likelihood estimation problem is

$$\mathbf{X}^\star_{ML} = \arg\min_{\mathbf{X} \in \mathbb{R}^{p\times C}} f(\mathbf{X})$$  

But does $f$ always attain its infimum? The following three questions examine this issue.

  

Explain the difference between infima and minima. Give an example of a convex function on $\mathbb{R}$ that does not attain its infimum.

  

Let $f: \mathcal{X} \to \mathbb{R}$ be a function defined on a domain $\mathcal{X}$. The infimum of $f$, denoted $\inf_{x \in \mathcal{X}} f(x)$, is the greatest lower bound of its image $f(\mathcal{X})$. Formally, a real number $m = \inf_{x \in \mathcal{X}} f(x)$ if $m \leq f(x)$ for all $x \in \mathcal{X}$, and for every $\epsilon > 0$, there exists $x \in \mathcal{X}$ such that $f(x) < m + \epsilon$.
A minimum is an infimum that is explicitly attained by the function within its domain. Specifically, the infimum is a minimum, denoted $\min_{x \in \mathcal{X}} f(x)$, if and only if $\exists x^* \in \mathcal{X}$ such that $f(x^*) = \inf_{x \in \mathcal{X}} f(x)$. Consequently, all minima are infima, but an infimum is a minimum strictly when the infimum is an element of the image set $f(\mathcal{X})$.

To illustrate a convex function on $\mathbb{R}$ that does not attain its infimum, consider $f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = e^x$. This function is strictly convex and bounded below by $0$. Evaluating the asymptotic behavior yields $\lim_{x \to -\infty} e^x = 0$, establishing that $\inf_{x \in \mathbb{R}} e^x = 0$. However, because $\nexists x \in \mathbb{R}$ such that $e^x = 0$, the infimum is not attained in the domain, and therefore $\min_{x \in \mathbb{R}} e^x$ is undefined.

  

### (d) (2 points)

Assume there exists $\mathbf{X}_0 \in \mathbb{R}^{p\times C}$ such that for all $i$,
$$\mathbf{a}_i^\top \mathbf{x}_{0, b_i} - \max_{k \neq b_i} \mathbf{a}_i^\top \mathbf{x}_{0,k} > 0.$$
This is called one-versus-all complete separation in multiclass settings. Give a geometric interpretation (e.g., for $p=2$) and explain why the name is appropriate.

  
<span style="color:rgb(255, 0, 0)">GEOMETRIC INTERPRETATION IS MISSING</span>

The name "one-versus-all complete separation" is appropriate because the inequality mathematically compares the score of the correct class ("one") against the maximum score of all alternative classes ("all"). 

  

  

### (e) (3 points)

Complete separation allows perfect classification of the training data. However, as you will show next, it prevents the existence of a finite maximum likelihood estimate.
In a one-versus-all complete separation setting (as in (d)), prove that $f$ does not attain its minimum. Hint: consider $f(\alpha \mathbf{X}_0)$ as $\alpha \to +\infty$ and compare it to $f(\mathbf{X}_0)$.

  
By assumption, $\mathbf{X}_0$ achieves complete separation, that is, for all $i$,
$$\mathbf{a}_i^\top \mathbf{x}_{0, b_i} - \max_{k \neq b_i} \mathbf{a}_i^\top \mathbf{x}_{0,k} > 0.$$
It is straightforward to check that if $\mathbf{X}_0$ achieves complete separation, then $\alpha \mathbf{X}_0$ achieves complete separation as well for any $\alpha > 0$:
$$\mathbf{a}_i^\top (\alpha\mathbf{x}_{0, b_i}) - \max_{k \neq b_i} \mathbf{a}_i^\top (\alpha\mathbf{x}_{0,k}) > 0 \iff \alpha \left(\mathbf{a}_i^\top \mathbf{x}_{0, b_i} - \max_{k \neq b_i} \mathbf{a}_i^\top \mathbf{x}_{0,k}\right) > 0 \iff \mathbf{a}_i^\top \mathbf{x}_{0, b_i} - \max_{k \neq b_i} \mathbf{a}_i^\top \mathbf{x}_{0,k} > 0.$$

To evaluate the behavior of the negative log-likelihood under complete separation, we substitute the scaled weight matrix $\alpha \mathbf{X}_0$ into the objective function:
$$f(\alpha \mathbf{X}_0) = \sum_{i=1}^n \left[ -\alpha \mathbf{a}_i^\top \mathbf{x}_{0,b_i} + \log \sum_{k=1}^C \exp(\alpha \mathbf{a}_i^\top \mathbf{x}_{0,k}) \right]$$

We can factor out the exponential term corresponding to the true class $b_i$ from the sum inside the logarithm:
$$\sum_{k=1}^C \exp(\alpha \mathbf{a}_i^\top \mathbf{x}_{0,k}) = \exp(\alpha \mathbf{a}_i^\top \mathbf{x}_{0,b_i}) \left( 1 + \sum_{k \neq b_i} \exp\big(\alpha (\mathbf{a}_i^\top \mathbf{x}_{0,k} - \mathbf{a}_i^\top \mathbf{x}_{0,b_i})\big) \right)$$
$$\log \sum_{k=1}^C \exp(\alpha \mathbf{a}_i^\top \mathbf{x}_{0,k}) = \alpha \mathbf{a}_i^\top \mathbf{x}_{0,b_i} + \log \left( 1 + \sum_{k \neq b_i} \exp\big(\alpha (\mathbf{a}_i^\top \mathbf{x}_{0,k} - \mathbf{a}_i^\top \mathbf{x}_{0,b_i})\big) \right)$$

Substituting this back into $f(\alpha \mathbf{X}_0)$, the $-\alpha \mathbf{a}_i^\top \mathbf{x}_{0,b_i}$ and $+\alpha \mathbf{a}_i^\top \mathbf{x}_{0,b_i}$ terms cancel out, yielding:
$$f(\alpha \mathbf{X}_0) = \sum_{i=1}^n \log \left( 1 + \sum_{k \neq b_i} \exp\big(\alpha (\mathbf{a}_i^\top \mathbf{x}_{0,k} - \mathbf{a}_i^\top \mathbf{x}_{0,b_i})\big) \right)$$

The complete separation assumption guarantees that $\alpha\mathbf{a}_i^\top \mathbf{x}_{0,b_i} > \alpha\mathbf{a}_i^\top \mathbf{x}_{0,k}$ for all $k \neq b_i$. Consequently, the exponent is strictly negative for all incorrect classes:
$$\alpha(\mathbf{a}_i^\top \mathbf{x}_{0,k} - \mathbf{a}_i^\top \mathbf{x}_{0,b_i}) < 0$$

As $\alpha \to +\infty$, the exponential of a strictly negative number approaches zero:
$$\lim_{\alpha \to +\infty} \exp\big(\alpha (\mathbf{a}_i^\top \mathbf{x}_{0,k} - \mathbf{a}_i^\top \mathbf{x}_{0,b_i})\big) = 0$$

Evaluating the limit for the entire function gives:
$$\lim_{\alpha \to +\infty} f(\alpha \mathbf{X}_0) = \sum_{i=1}^n \log(1 + 0) = 0$$
This shows that the infimum of $f(\mathbf{X})$ is $0$. 

However, for any finite weight matrix $\mathbf{X} \in \mathbb{R}^{p \times C}$, the sum $\sum_{k \neq b_i} \exp(\mathbf{a}_i^\top \mathbf{x}_k - \mathbf{a}_i^\top \mathbf{x}_{b_i})$ is strictly greater than $0$ because the exponential function never evaluates to zero. Therefore, $f(\mathbf{X}) > 0$ $\forall \mathbf{X} \in \mathbb{R}^{p \times C}$. Because the function can approach $0$ asymptotically but never reach it for any finite parameters, it does not attain its minimum.

  



  

### (f) (2 points)

We resolve this issue by adding a regularizer. Consider the regularized function:
$$f_\mu(\mathbf{X}) = f(\mathbf{X}) + \frac{\mu}{2} \Vert{}\mathbf{X}\Vert{}_F^2, \quad \mu > 0.$$

Show that the gradient with respect to $\mathbf{X}$ of $f_\mu$ can be expressed as:
$$\nabla_{\mathbf{X}} f_\mu(\mathbf{X}) = \sum_{i=1}^n \mathbf{a}_i \big( \mathbf{p}_i - \mathbf{e}_{b_i} \big)^\top + \mu \mathbf{X},$$
where $\mathbf{e}_{b_i} \in \mathbb{R}^C$ is the one-hot vector for class $b_i$, $\mathbf{p}_i \in \mathbb{R}^C$ has entries $p_{i,c} = \mathbb{P}(b_i=c\mid \mathbf{a}_i)$ under the softmax model, and $\mathbf{a}_i(\mathbf{p}_i - \mathbf{e}_{b_i})^\top\in\mathbb{R}^{p\times C}$ denotes the outer product.





Solution:


By the linearity of the gradient operator, we can separate the derivative of the regularized objective function $f_\mu(\mathbf{X})$ into the gradient of the negative log-likelihood $f(\mathbf{X})$ and the gradient of the regularization term, yielding the equation:

$$\nabla_{\mathbf{X}} f_\mu(\mathbf{X}) = \nabla_{\mathbf{X}} f(\mathbf{X}) + \nabla_{\mathbf{X}} \left( \frac{\mu}{2} \Vert{}\mathbf{X}\Vert{}_F^2 \right)$$
By definition, the Frobenius norm of a matrix $\mathbf{X} \in \mathbb{R}^{p \times C}$ is given by:$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^p \sum_{j=1}^C X_{ij}^2}$$or equivalently as:
$$\Vert\mathbf{X}\Vert_F = \sqrt{\text{Tr}(\mathbf{X}^\top \mathbf{X})}$$

So:
$$\Vert\mathbf{X}\Vert_F^2 = \text{Tr}(\mathbf{X}^\top \mathbf{X})$$
Therefore, the gradient of the regularization term evaluates to:
$$\nabla_{\mathbf{X}} \left( \frac{\mu}{2} \|\mathbf{X}\|_F^2 \right) = \frac{\mu}{2}  \text{Tr}(\nabla_{\mathbf{X}}\mathbf{X}^\top \mathbf{X}) = \mu \mathbf{X}$$

$\mathbf{X}$ is composed of $C$ column vectors $\mathbf{x}_c \in \mathbb{R}^p$:
$$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \dots & \mathbf{x}_C \end{bmatrix}$$
As showed in point a) the negative log-likelihood is given by:
$$f(\mathbf{X}) = \sum_{i=1}^n \left[ -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right]$$
And define: $$f(\mathbf{X}) = \sum_{i=1}^n f_i(\mathbf{X})$$
where
$$f_i(\mathbf{X}) = -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)$$
The full gradient matrix $\nabla_{\mathbf{X}} f(\mathbf{X})$ is given by:
$$\nabla_{\mathbf{X}} f(\mathbf{X}) = \begin{bmatrix} \frac{\partial f}{\partial \mathbf{x}_1} & \frac{\partial f}{\partial \mathbf{x}_2} & \dots & \frac{\partial f}{\partial \mathbf{x}_C} \end{bmatrix}$$
Where each component $\frac{\partial f}{\partial \mathbf{x}_c}$ is a $p$ dimensional column vector containing the partial derivatives of $f$with respect to each scalar element of $\mathbf{x}_c$:
$$\frac{\partial f}{\partial \mathbf{x}_c} = \begin{bmatrix} \frac{\partial f}{\partial x_{1,c}} \\ \frac{\partial f}{\partial x_{2,c}} \\ \vdots \\ \frac{\partial f}{\partial x_{p,c}} \end{bmatrix}$$
The derivative of $-\mathbf{a}_i^\top \mathbf{x}_{b_i}$ with respect to $\mathbf{x}_c$ is $-\mathbf{a}_i$ if the target class $b_i$ is equal to $c$, and 0 otherwise. We can write this compactly using the the one-hot vector $\mathbf{e}_{b_i}$, giving:
$$\frac{\partial}{\partial \mathbf{x}_c} \left( -\mathbf{a}_i^\top \mathbf{x}_{b_i} \right) = -e_{b_i,c} \mathbf{a}_i$$

Instead using the  the chain rule to differentiate the summation. Because the only term in the sum that depends on $\mathbf{x}_c$ is $\exp(\mathbf{a}_i^\top \mathbf{x}_c)$, the derivative becomes:
$$\frac{\partial}{\partial \mathbf{x}_c} \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) = \frac{1}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)} \cdot \exp(\mathbf{a}_i^\top \mathbf{x}_c) \cdot \mathbf{a}_i$$
Therefore:
$$\frac{\partial}{\partial \mathbf{x}_c} \left( -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right) = -e_{b_i,c} \mathbf{a}_i + \left( \frac{\exp(\mathbf{a}_i^\top \mathbf{x}_c)}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)} \right) \mathbf{a}_i$$Note that the softmax probability for class $c$, denoted as $p_{i,c}$ is given by:
$$p_{i,c} = \frac{\exp(\mathbf{a}_i^\top \mathbf{x}_c)}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)}$$

Substituting $p_{i,c}$ back into the equation allows us to simplify the final gradient for sample $i$ with respect to $\mathbf{x}_c$:
$$\frac{\partial}{\partial \mathbf{x}_c} \left( -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right) = p_{i,c} \mathbf{a}_i - e_{b_i,c} \mathbf{a}_i = \mathbf{a}_i (p_{i,c} - e_{b_i,c})$$

To assemble the gradient matrix for a single sample $i$ with respect to the entire matrix $\mathbf{X}$, we concatenate the column gradients for all classes $c \in \{1, \dots, C\}$, that is:
$$\nabla_{\mathbf{X}} f_i(\mathbf{X}) = \begin{bmatrix} \frac{\partial}{\partial \mathbf{x}_1} \left( -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right) & \dots & \frac{\partial}{\partial \mathbf{x}_C} \left( -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k) \right) \end{bmatrix}$$

Substituting the simplified column derivatives derived earlier, this evaluates to:
$$\nabla_{\mathbf{X}} f_i(\mathbf{X}) = \begin{bmatrix} \mathbf{a}_i (p_{i,1} - e_{b_i,1}) & \mathbf{a}_i (p_{i,2} - e_{b_i,2}) & \dots & \mathbf{a}_i (p_{i,C} - e_{b_i,C}) \end{bmatrix}$$
$$= \mathbf{a}_i \begin{bmatrix} (p_{i,1} - e_{b_i,1}) & (p_{i,2} - e_{b_i,2}) & \dots & (p_{i,C} - e_{b_i,C}) \end{bmatrix}=\mathbf{a}_i (\mathbf{p}_i - \mathbf{e}_{b_i})^\top$$

By the linearity of the derivative, we sum the gradient for the single sample $\nabla_{\mathbf{X}} f_i(\mathbf{X})$ over all $n$ independent samples to obtain the full gradient of the negative log-likelihood $f(\mathbf{X})$:

$$\nabla_{\mathbf{X}} f(\mathbf{X}) = \sum_{i=1}^n \nabla_{\mathbf{X}} f_i(\mathbf{X}) = \sum_{i=1}^n \mathbf{a}_i (\mathbf{p}_i - \mathbf{e}_{b_i})^\top$$

Next, we recall that the full regularized objective function $f_\mu(\mathbf{X})$ is the sum of this negative log-likelihood and the Frobenius norm regularization term:

$$f_\mu(\mathbf{X}) = f(\mathbf{X}) + \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2$$

Applying the gradient operator with respect to $\mathbf{X}$ to both sides gives:
$$\nabla_{\mathbf{X}} f_\mu(\mathbf{X}) = \nabla_{\mathbf{X}} f(\mathbf{X}) + \nabla_{\mathbf{X}} \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right)$$
$$\nabla_{\mathbf{X}} f_\mu(\mathbf{X}) = \sum_{i=1}^n \mathbf{a}_i \big( \mathbf{p}_i - \mathbf{e}_{b_i} \big)^\top + \mu \mathbf{X}$$



  

### (g) (4 points)

Use row-wise vectorization of $\mathbf{X}$, obtained by stacking its rows one after another into a vector in $\mathbb{R}^{pC}$. Show that the Hessian of $f_\mu$ in these coordinates can be written as:
$$\nabla^2 f_\mu(\mathbf{X}) = \sum_{i=1}^n (\mathbf{a}_i\mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big) + \mu \mathbf{I}_{pC},$$
where $\otimes$ is the Kronecker product, and $\operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top$ is the softmax Jacobian, which is positive semidefinite.





**Solution:**

Let $\mathbf{X} \in \mathbb{R}^{pC}$.

By the linearity of the Hessian operator:
$$\nabla^2 f_\mu(\mathbf{X}) = \nabla^2 f(\mathbf{X}) + \nabla^2 \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right)$$

As established in point (f), the column gradient is $\nabla_{\mathbf{X}} \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right) = \mu \mathbf{X}$, meaning the row gradient is $\nabla_{\mathbf{X}^\top} \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right) = \mu \mathbf{X}^\top$. Therefore, applying the Hessian operator yields:
$$\nabla^2 \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right) = \nabla_{\mathbf{X}} \nabla_{\mathbf{X}^\top} \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right) = \nabla_{\mathbf{X}} (\mu \mathbf{X}^\top)$$


Thus, the Hessian matrix of the regularizer is simply the identity matrix scaled by $\mu$:
$$\nabla^2 \left( \frac{\mu}{2} \Vert\mathbf{X}\Vert_F^2 \right) = \mu \mathbf{I}_{pC}$$
To derive the Hessian using only standard partial derivatives, we compute the second derivative of the negative log-likelihood for a single sample $i$ with respect to individual scalar elements of $\mathbf{X}$.

  

Let $X_{jc}$ be the scalar element in the $j$-th row and $c$-th column of $\mathbf{X}$.
From point (f):
$$f(\mathbf{X}) = \sum_{i=1}^n f_i(\mathbf{X})$$
where
$$f_i(\mathbf{X}) = -\mathbf{a}_i^\top \mathbf{x}_{b_i} + \log \sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)$$
then the partial derivative of the negative log-likelihood $f_i$ with respect to $X_{jc}$ is:
$$\frac{\partial f_i}{\partial X_{jc}} = a_{i,j} (p_{i,c} - e_{b_i,c})$$
where $a_{i,j}$ is the $j$-th element of the feature vector $\mathbf{a}_i$.

To find the elements of the Hessian matrix, we take the second partial derivative with respect to another arbitrary element $X_{kd}$:
$$\frac{\partial^2 f_i}{\partial X_{jc} \partial X_{kd}} = \frac{\partial}{\partial X_{kd}} \big[ a_{i,j} (p_{i,c} - e_{b_i,c}) \big] = a_{i,j} \frac{\partial p_{i,c}}{\partial X_{kd}}$$

By the chain rule, we can differentiate $p_{i,c}$ with respect to $X_{kd}$:
$$\frac{\partial p_{i,c}}{\partial X_{kd}} = \frac{\partial}{\partial X_{kd}}\frac{\exp(\mathbf{a}_i^\top \mathbf{x}_c)}{\sum_{k=1}^C \exp(\mathbf{a}_i^\top \mathbf{x}_k)}=\sum_{m=1}^C \frac{\partial p_{i,c}}{\partial z_{i,m}} \frac{\partial z_{i,m}}{\partial X_{kd}}$$
where $z_{i,m} = \mathbf{a}_i^\top \mathbf{x}_m = \sum_{l=1}^p a_{i,l} X_{l,m}$.

Notice that $z_{i,m}$ only depends on the elements in the **$m$-th column** of the matrix $\mathbf{X}$. But we are differentiating with respect to $X_{kd}$, which is an element in the **$d$-th column** of $\mathbf{X}$.

So:
If $m \neq d$: The formula for $z_{i,m}$ does not contain the variable $X_{kd}$ at all. Therefore, $\frac{\partial z_{i,m}}{\partial X_{kd}} = 0$.
If $m = d$: The formula is $z_{i,d} = \sum_{l=1}^p a_{i,l} X_{l,d}$. The derivative of this with respect to $X_{kd}$ is just the coefficient $a_{i,k}$.


Because of this:
$$\sum_{m=1}^C \frac{\partial p_{i,c}}{\partial z_{i,m}} \frac{\partial z_{i,m}}{\partial X_{kd}} = \frac{\partial p_{i,c}}{\partial z_{i,1}}(0) + \dots + \frac{\partial p_{i,c}}{\partial z_{i,d}}(a_{i,k}) + \dots + \frac{\partial p_{i,c}}{\partial z_{i,C}}(0)=\frac{\partial p_{i,c}}{\partial z_{i,d}}a_{i,k}$$

Therefore:
$$\frac{\partial p_{i,c}}{\partial X_{kd}} = \frac{\partial p_{i,c}}{\partial z_{i,d}} a_{i,k}$$


Now we need  to evaluate:$$\frac{\partial p_{i,c}}{\partial z_{i,d}}$$
Let $c = d$

Differentiating with respect to $z_{i,c}$ directly transforms the quotient rule application into the factored probability:
$$\frac{\partial p_{i,c}}{\partial z_{i,c}} = \frac{\exp(z_{i,c}) \sum_{m=1}^C \exp(z_{i,m}) - \exp(z_{i,c})^2}{\left( \sum_{m=1}^C \exp(z_{i,m}) \right)^2} = p_{i,c}(1 - p_{i,c})$$

Let $c \neq d$

Differentiating with respect to $z_{i,d}$ zeroes out the numerator derivative, reducing the quotient application directly to the negative product of the probabilities:
$$\frac{\partial p_{i,c}}{\partial z_{i,d}} = \frac{0 - \exp(z_{i,c})\exp(z_{i,d})}{\left( \sum_{m=1}^C \exp(z_{i,m}) \right)^2} = -p_{i,c} p_{i,d}$$

We can combine the two cases into a single, compact expression using  $\delta_{cd}$ which evaluates to $1$ if
$c=d$ and $0$ if $c \neq d$:
$$\frac{\partial p_{i,c}}{\partial z_{i,d}} = p_{i,c}(\delta_{cd} - p_{i,d})$$

Substituting these back into the chain rule gives:
$$\frac{\partial p_{i,c}}{\partial X_{kd}} = p_{i,c} (\delta_{cd} - p_{i,d}) a_{i,k}$$

Inserting this into our second derivative equation, we obtain the scalar element of the Hessian:
$$\frac{\partial^2 f_i}{\partial X_{jc} \partial X_{kd}} = a_{i,j} a_{i,k} p_{i,c} (\delta_{cd} - p_{i,d})$$

Notice that this scalar value is exactly the product of two distinct terms:

1. **$a_{i,j} a_{i,k}$**:  Which is the $(j, k)$-th element of the $p \times p$ outer product matrix $\mathbf{a}_i \mathbf{a}_i^\top$.
  
2. $p_{i,c} (\delta_{cd} - p_{i,d})$, which forms the $(c, d)$-th entry of the $C \times C$ softmax Jacobian matrix $\operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top$. This structure is immediately clear by expanding the matrix expression:$$\operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top = \begin{bmatrix} p_{i,1} & 0 & \dots & 0 \\ 0 & p_{i,2} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & p_{i,C} \end{bmatrix} - \begin{bmatrix} p_{i,1}^2 & p_{i,1}p_{i,2} & \dots & p_{i,1}p_{i,C} \\ p_{i,2}p_{i,1} & p_{i,2}^2 & \dots & p_{i,2}p_{i,C} \\ \vdots & \vdots & \ddots & \vdots \\ p_{i,C}p_{i,1} & p_{i,C}p_{i,2} & \dots & p_{i,C}^2 \end{bmatrix}$$
  

Therefore, mapping all scalar second derivatives $\frac{\partial^2 f_i}{\partial X_{jc} \partial X_{kd}}$ into a $pC \times pC$ matrix using the Kronecker product yields:
$$\nabla^2 f_i(\mathbf{X}) = (\mathbf{a}_i \mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big)$$

By the linearity of the Hessian operator, we sum this over all $n$ samples to get the Hessian of the negative log-likelihood:
$$\nabla^2 f(\mathbf{X}) = \sum_{i=1}^n (\mathbf{a}_i \mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big)$$

Finally, adding the Hessian of the regularization term ($\mu \mathbf{I}_{pC}$) gives the complete regularized Hessian:
$$\nabla^2 f_\mu(\mathbf{X}) = \sum_{i=1}^n (\mathbf{a}_i\mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big) + \mu \mathbf{I}_{pC}$$






  

  

### (h) (2 points)

Show that $f_\mu$ is $\mu$-strongly convex.

  


Solution:

  
By the lemma seen in class, a twice-differentiable convex function $f: \mathcal{Q} \to \mathbb{R}$, with $\mathcal{Q} \subseteq \mathbb{R}^{pC}$ and $f \in \mathcal{C}^2(\mathcal{Q})$, is $\mu$-strongly convex if and only if its Hessian satisfies $\nabla^2 f(\mathbf{X}) \succeq \mu \mathbf{I}$ for all $\mathbf{X} \in \mathcal{Q}$. 

From part (g), the Hessian of the regularized objective function is given by:
$$\nabla^2 f_\mu(\mathbf{X}) = \sum_{i=1}^n (\mathbf{a}_i\mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big) + \mu \mathbf{I}_{pC}$$
The outer product matrix $\mathbf{a}_i\mathbf{a}_i^\top$ is positive semidefinite because for any arbitrary vector $\mathbf{v}$, the quadratic form yields:
$$\mathbf{v}^\top (\mathbf{a}_i\mathbf{a}_i^\top) \mathbf{v} = (\mathbf{a}_i^\top \mathbf{v})^2 \geq 0$$$\operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top$ represents the variance-covariance matrix of a multinomial distribution that is
$\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{e}_{b_i})$,  and is therefore a positive semidefinite matrix, proof of this fact is attached below.

A standard property of the Kronecker product is that the product of two positive semidefinite matrices is positive semidefinite. Consequently:
$$(\mathbf{a}_i\mathbf{a}_i^\top) \otimes \big( \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top \big) \succeq 0$$Because the sum of positive semidefinite matrices is also positive semidefinite, summing over all $n$ samples guarantees that the entire unregularized Hessian $\nabla^2 f(\mathbf{X})$ is positive semidefinite ($\nabla^2 f(\mathbf{X}) \succeq 0$).

Substituting this established bound back into the full Hessian expression yields:
$$\nabla^2 f_\mu(\mathbf{X}) = \nabla^2 f(\mathbf{X}) + \mu \mathbf{I}_{pC} \succeq \mu \mathbf{I}_{pC}$$
Because $\nabla^2 f_\mu(\mathbf{X}) \succeq \mu \mathbf{I}_{pC}$, the function $f_\mu$ satisfies the condition for $\mu$-strong convexity.





**Proof of $\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{e}_{b_i})= \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top$**

We represent the target label as a random vector $\mathbf{e}_{b_i} \in \mathbb{R}^C$, where $\mathbf{e}_{b_i}$ is the one-hot vector for class $b_i$. For any given sample, exactly one element of $\mathbf{e}_{b_i}$ is $1$ and the remaining $C - 1$ elements are $0$. Let $e_{i,c}$ denote the $c$-th component of this vector, meaning $e_{i,c} \in \{0, 1\}$.

Under our softmax model, $\mathbf{e}_{b_i}$ follows a multinomial (categorical) distribution with a single trial and probability vector $\mathbf{p}_i$.

The expected value of any component $e_{i,c}$ is the probability that $b_i = c$:
$$\mathbb{E}[e_{i,c}] = 1 \cdot p_{i,c} + 0 \cdot (1 - p_{i,c}) = p_{i,c}$$
Thus, the mean vector is $\mathbb{E}[\mathbf{e}_{b_i}] = \mathbf{p}_i$.

The variance-covariance matrix $\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{e}_{b_i})$ is populated by the variances of each component on the diagonal and the covariances between components on the off-diagonals.

  

For the diagonal elements:
$$\operatorname{Var}(e_{i,c}) = \mathbb{E}[e_{i,c}^2] - (\mathbb{E}[e_{i,c}])^2 = p_{i,c} - p_{i,c}^2 = p_{i,c}(1 - p_{i,c})$$

Considering the off-diagonal elements for any two distinct classes $c \neq k$, the covariance is:
$$\operatorname{Cov}(e_{i,c}, e_{i,k}) = \mathbb{E}[e_{i,c} e_{i,k}] - \mathbb{E}[e_{i,c}]\mathbb{E}[e_{i,k}]$$
Because the label $b_i$ can only take one class value at a time, the vector $\mathbf{e}_{b_i}$ cannot have $1$s in both position $c$ and position $k$ simultaneously. This means their product $e_{i,c} e_{i,k}$ is strictly $0$ in all cases. Therefore, $\mathbb{E}[e_{i,c} e_{i,k}] = 0$.

Substituting the expectations yields:
$$\operatorname{Cov}(e_{i,c}, e_{i,k}) = 0 - p_{i,c}p_{i,k} = -p_{i,c}p_{i,k}$$
Therefore:
$$\boldsymbol{\Sigma} = \operatorname{Var}(\mathbf{e}_{b_i})= \operatorname{Diag}(\mathbf{p}_i) - \mathbf{p}_i\mathbf{p}_i^\top$$


  

  

### (i) (2 points)

Is it possible for a strongly convex function to not attain its minimum? Justify your reasoning (you may assume the domain is $\mathbb{R}^{p\times C}$).

  

_Write your answer here._

  















### (j) (4 points for all three questions)

We will now show that $f_\mu$ is smooth, i.e., $\nabla f_\mu$ is L-Lipschitz with respect to the Frobenius norm, with a simple conservative bound:

  

$$L = \Vert{}\mathbf{A}\Vert{}_F^2 + \mu.$$

  

where:

  

$$\mathbf{A} = \begin{bmatrix} \leftarrow & \mathbf{a}_1^\top & \rightarrow \\ \leftarrow & \mathbf{a}_2^\top & \rightarrow \\ & \ldots & \\ \leftarrow & \mathbf{a}_n^\top & \rightarrow \\ \end{bmatrix}.$$

  

_(You may use that the operator norm of the softmax Jacobian is bounded by $1/2$, and a looser bound $\le 1$ is acceptable for grading.)_

  

  

Hint: check the properties of the spectral norm with respect to dot product, Kronecker product, and outer product.

  

**(j-1)** Show that $\lambda_{\max}(\mathbf{a}_i\mathbf{a}_i^\top) = \Vert{}\mathbf{a}_i\Vert{}_2^2$, where $\lambda_{\max}(\cdot)$ denotes the largest eigenvalue.

  

_Write your answer here._

  

  

**(j-2)** Using the derived Hessian, show that $\lambda_{\max}(\nabla^2 f_\mu(\mathbf{X})) \leq \sum_{i=1}^{n} \Vert{}\mathbf{a}_i\Vert{}_2^2 + \mu$.

  

_Write your answer here._

  

  

**(j-3)** Conclude that $f_\mu$ is $L$-smooth for $L = \Vert{}\mathbf{A}\Vert{}_F^2 + \mu$.

  

_Write your answer here._

  

  

### (l) (3 points)

KL divergence and NLL. Let $q_i=(q(c\mid\mathbf{a}_i))_{c=1}^C$ be the true conditional label distribution and $\mathbf{p}_i$ the model softmax distribution. Write $\mathrm{KL}(q_i\,\Vert{}\,\mathbf{p}_i)$ and show that minimizing its average over the samples is equivalent to minimizing the expected average negative log-likelihood, where the expectation is over labels drawn from $q$ with the features held fixed. Explain how $f(\mathbf{X})/n$ from (a) estimates this expected loss.

  

_Write your answer here._

  

  

### The Regularized Estimator

The unregularized maximum likelihood problem may have no finite minimizer. Adding a quadratic regularizer with $\mu>0$ gives a unique regularized estimate, defined by the smooth, strongly convex problem:

  

$$\mathbf{X}^\star = \arg\min_{\mathbf{X} \in \mathbb{R}^{p\times C}} f(\mathbf{X}) + \frac{\mu}{2}\Vert{}\mathbf{X}\Vert{}_F^2.$$

  

  

  

## 2. Binary logistic regression (specialization for Part 2)

  

  

While this part analyzed the multiclass (softmax) setting, in the next exercise we will continue under the simplified two-class case.

  

Let labels be $b_i \in \{-1, +1\}$, features $\mathbf{a}_i \in \mathbb{R}^p$, and weight vector $\mathbf{x} \in \mathbb{R}^p$. Define the sigmoid:

  

$$\sigma(t) = \frac{1}{1+e^{-t}}.$$

  

Model the conditional distribution as:

  

$$\mathbb{P}(b_i = j \mid \mathbf{a}_i) = \sigma\big(j\, \mathbf{a}_i^\top \mathbf{x}\big), \quad j \in \{-1,+1\}.$$

  

The likelihood over i.i.d. samples $\{(\mathbf{a}_i, b_i)\}_{i=1}^n$ is:

  

$$\mathcal{L}(\mathbf{x}) = \prod_{i=1}^n \sigma\big(b_i\, \mathbf{a}_i^\top \mathbf{x}\big),$$

  

so the negative log-likelihood is:

  

$$f(\mathbf{x}) = -\log \mathcal{L}(\mathbf{x}) = \sum_{i=1}^n \log\big(1 + e^{-b_i\, \mathbf{a}_i^\top \mathbf{x}}\big).$$

  

### (m) (2 points)

Show that the gradient of the negative log-likelihood is the standard binary logistic regression gradient:

  

$$\nabla f(\mathbf{x}) = \sum_{i=1}^n \big(-b_i\, \sigma(-b_i\, \mathbf{a}_i^\top \mathbf{x})\big)\, \mathbf{a}_i.$$

  

(Hint: use the chain rule and $\sigma'(t) = \sigma(t)\big(1-\sigma(t)\big)$.)

  

We will use this binary formulation in Part 2 - First order methods.

  

_Write your answer here._
