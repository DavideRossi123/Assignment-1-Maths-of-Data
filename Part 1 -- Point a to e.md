
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

  


