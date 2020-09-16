---
layout: post
usemathjax: true
title:  "Support Vector Machines"
date:   2020-09-16 14:30:42 +0100
categories: jekyll update
---

# Support Vector Machines

The following notes are a pedagogical synthesis of material in good quality sources on machine learning to aid my understanding of support vector machines.

The core sources we rely on are graduate-level university lectures, reputable introductory textbooks, the papers of those who pioneered these techniques, as well as well-known tutorials. On the whole, these sources were selected to the extent that they are authoritative, clear and rigorous. 

The purpose of the synthesis is to bring together insights across these materials in an effort to develop an understanding that is robust to differences in notation; that integrates the various theoretical perspectives underlying this technique; that acts as an elementary base from which to deepen my knowledge; and to distil intuitions that are helpful to understanding.

Significantly, those insights, interpretations, intuitions which feature repeatedly across sources are included. The author of this believes that a survey of this kind allows a mapping of what is largely agreed upon in the broader community, as well as points which are more contentious and which do not enjoy the same degree of consensus.

### Motivation

We will begin our discussion of the support vector machine in context of binary linear classification. Extensions such as overlapping classes, non-linear decision boundaries, and multi-class classification, are discussed later on.

Given that we have a set of labelled training data, we would like to use this information to define a classification rule. With this new classification rule, we can then say whether a new unlabelled test data point belongs to either of two classes.

We assume that the data is linearly separable, so that it is possible to define a classification rule that enables us to say, for every data point, whether a point belongs to one class or another.

A visualisation will help further introduce the geometrical principles underlying its conception:

In this context, there remains an outstanding issue. That is, even if we assume linear separability, it is evident that there are multiple decision boundaries we can construct to classify the data. 

How do we choose which decision boundary is appropriate?

### Maximum-margin classification

The support vector machine supplies an optimality criterion by which we can choose from a number of linear classifiers – the ‘best’ linear classifier from a competing set of linear classifiers is the one that maximises the margin. 

Informally, rather than drawing a line of zero-width to separate the classes, we draw around each line a margin of some width, up to the nearest point. According to our criterion, the line that maximises the margin is the optimal classifier. 

Intuitively, this captures the essence of the support vector machine, and much of the mathematics invoked in its construction serves to formalise this according to systematic principles. At an appropriate level, understanding the mechanics of the support vector machine requires some exposition using the geometry of vector algebra, as well as results in constrained convex optimisation.

### Formal definitions
*Get the skeleton done up, then polish for a target audience as necessary

We are given a set of *N* input vectors $\mathbf{x}_1$, $\mathbf{x}_2$, ..., $\mathbf{x}_N$, and corresponding class labels $t_1$, $t_2$, ..., $t_N$ where each $\mathbf{x}_n \in \mathbb{R} ^D$ and each $t_n \in \left \{-1, +1 \right\}$.

The entire sequence $\left \{ \left(\mathbf{x}_n, t_n \right)\right\}_{n=1}^N$ constitutes our *training data*.

We define a *linear decision function* to be of the form:

$$ y(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}) + b = \sum_{j=1}^{M} w_j \phi_j(\mathbf{x}) + b$$

where $\mathbf{w}\in\mathbb{R}^M$ is an $M$-dimensional vector, and $b \in\mathbb{R}$ is a scalar.

$\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^M$ is an $M$-dimensional vector, whose elements consist of $M$ fixed basis functions $\left(\phi_1, \phi_2,...,\phi_M \right)^T$, with each element $\phi_j = \phi_j(\mathbf{x})$.

These basis functions are often nonlinear functions of the input vectors, and can be thought of as a pre-processing or feature extraction step of the original input variables. 

The decision function is *linear* because it is a linear function of the *parameters* $(\mathbf{w}, b)$

A *binary linear classifier* is a function of the form:

$$f(x) = sign(y(\mathbf{x})) =  sign(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}) + b) $$

Two sets, $A,B\subset\mathbb{R}^M$ are *linearly separable* if:

$$y(\mathbf{x}) = \mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}) + b = \left\{
 \begin{array}{ll}
      > 0 & \quad if x \in A (e.g. class +1) \\
      < 0 & \quad if x \in B (e.g. class -1)
 \end{array}
\right.
$$

A *hyperplane* $H$ in $\mathbb{R}^M$ is a linear subspace of dimension $(M - 1)$. It can be repesented by a vector $\mathbf{w}$ as follows:

$$ H = \{\mathbf{x} \in \mathbb{R}^M \mid \mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}) = 0\} $$

An *affine hyperplane* is a hyperplane that has been shifted by a scalar $b$, and is represented/parametrised with a vector $\mathbf{w}$ and scalar $b$ as follows:

$$ H = \{\mathbf{x} \in \mathbb{R}^M \mid \mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}) + b = 0\} $$

In this context, binary linear classification amounts to using our training data to provide estimates of $\mathbf{\hat{w}}$ and $\hat{b}$. 

Each estimate parametrises an affine hyperplane $H$, and following our exposition earlier, there will be multiple estimates.

Now that we have formally defined an affine hyperplane, we can now formalise what a margin is mathematically.

The perpendicular distance of an arbitrary input vector $\mathbf{x}$ from the affine hyperplane $H$ is given by:

$$ \frac{\left \lvert y(\mathbf{x}) \right \rvert}{\lVert \mathbf{w} \rVert_2} $$

As we are only interested in solutions for which all training data points are correctly classified, we require that $t_n y(\mathbf{x}_n) > 0$ for all $n$. Note that this condition arises from the coincidence of signs.

The distance of an arbitrary point $\mathbf{x_n}$ to the hyperplane is then given by:

$$ \frac{t_n y(\mathbf{x}_n)}{\lVert \mathbf{w} \rVert_2} =  \frac{t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b)}{\lVert \mathbf{w} \rVert_2}  $$

The *margin* is given by the smallest distance between the decision boundary and any of the N data points; or the perpendicular distance between the closest point and the decision boundary.

Therefore we wish to find values of our parameters $\mathbf{w}$ and $b$ by maximising this quantity.

$$\underset{\mathbf{w} , b}{\arg\max}
\left\{
 \begin{array}{ll}
      \frac{1}{\lVert \mathbf{w} \rVert_2} \underset{N}{\min}[t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b)]
 \end{array}
\right\}
$$

where the factor $\frac{1}{\lVert w \rVert_2}$ is taken outside the optimisation over $n$ because $\mathbf{w}$ does not depend on $n$

[SHOW THIS - Combining two constraints into one]

Much of the field of *convex optimisation* is concerned with transforming difficult problems for which we have not yet developed tools to solve into problems which are very much amenable to existing tools.

There are a number of preparatory steps that we need to take to arrive at the well known constrained convex optimisation problem underlying the support vector machine.

Note that the distance from any point $\mathbf{x}_n$ to the affine hyperplane is invariant to a proportional rescaling $\mathbf{w} \to \kappa \mathbf{w}$ and $b \to \kappa b$.

Using this, for *only* the point that is closest to the hyperplane, we set

$$t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) = 1$$

Only the point closest to the hyperplane will satisfy this condition.

All the data points, together with the point closest to the hyperplane will satisfy the N inequality constraints:

$$ t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) \geq 1  \quad \forall n = 1,..., N $$

This is known in the literature as the *canonical representation* of the decision hyperplane.

For data points where the constraint holds with equality, then the constraint is known as *active* or *tight*.

For data points where the constraint does not hold with equality, then the constraint is known as *inactive* or *slack*.

There will always be one active constraint, as there will always be one closest point by definition. 

Once we have found parameter estimates $\mathbf{\hat w}$ and $\hat b$ there will be at least two active constraints. (ELABORATE WHY)

As a further preparatory step, we note that maximising $\frac{1}{\lVert \mathbf{w} \rVert_2^2}$ is equivalent to minimising $\lVert \mathbf{w} \rVert_2$. 

Additionally, we introduce a computational convenience factor of $\frac{1}{2}$, yielding the following constrained convex optimisation problem:

$$
\underset{\mathbf{w}, b}{\arg \min} \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 \quad \text{subject to} \quad t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) \geq 1  \quad \forall n = 1,..., N
$$

More specifically, it is a *quadratic programming* problem. We have to find values of the parameters, i.e. estimates that minimise the *quadratic function* in $\mathbf{w}$, subject to N *linear inequality constraints*.

### Constrained optimisation using Lagrange multipliers

Constrained optimisation problems can be handled using the method of *Lagrange multipliers*.

Define $N$ Lagrange multipliers $\alpha_1, \alpha_2,..., \alpha_N$corresponding to N inequality constraints, and stack them into an $N$-dimensional vector $ \mathbf{a} = (\alpha_1,\alpha_2,...,\alpha_N)^T$.

Each Lagrange multiplier $\alpha_n \geq 0$

We then define the following *Lagrangian function*:

$$\begin{align}
\mathit{L}(\mathbf{w}, b, \mathbf{a}) \quad & = \quad \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 \ - \ \sum_{n=1}^N\alpha_n \{t_n(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 \} \\
&= \quad \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 \ - \ \sum_{n=1}^N \alpha_n t_n(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) + \sum_{n=1}^N \alpha_n 
\end{align}$$

We minimise the *Lagrangian primal* with respect to $\mathbf{w}$ and $b$:

$$\begin{align}
& \underset{\mathbf{w}, b}{\min} \quad \mathit{L}_\mathit{P}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 \ - \ \sum_{n=1}^N \alpha_n t_n(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) \ + \ \sum_{n=1}^N \alpha_n  \\
\end{align}$$

For constraints of the form $c_i \geq 0$, we require that the Lagrange multipliers be positive and we subtract them from the objective. 
[BURGES explanation]
[POSITIVITY of Lagrange multipliers]

Setting the gradient and derivative of the Lagrangian function $\mathit{L}$ with respect to $\mathbf{w}$ and $b$ equal to 0 respectively, we have:

$$\nabla_{\mathbf{w}} \ \mathit{L} = \mathbf{0} \implies \mathbf{w} = \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n) $$

$$\frac{\partial\mathit{L}}{\partial b} = 0 \implies \sum_{n=1}^N \alpha_n t_n = 0$$

Substituting for $\mathbf{w}$ yields:

[RELATION OF THE PRIMAL TO THE DUAL - dual as a lower bound on the primal problem for any feasible point]
[Reasons for the dual - efficient algorithm]
[Convex hull interpretation]

$$\begin{align}
\mathit{L}_\mathit{D}(\mathbf{a}) \quad & = \quad \frac{1}{2} \left \lVert \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n)  \right\rVert_2^2 \ - \ \sum_{n=1}^N\alpha_n \{t_n(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 \} \\
&= \quad \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^M \alpha_n \alpha_m t_n t_m \boldsymbol{\phi}(\mathbf{x}_n) ^T \boldsymbol{\phi}(\mathbf{x}_m) - \sum_{n=1}^N \sum_{m=1}^M \alpha_n \alpha_m t_n t_m \boldsymbol{\phi}(\mathbf{x}_n) ^T \boldsymbol{\phi}(\mathbf{x}_m) - b \sum_{n=1}^N \alpha_n t_n + \sum_{n=1}^N \alpha_n
\end{align}$$

As $\sum_{n=1}^N \alpha_n t_n = 0$, the term $b\sum_{n=1}^N \alpha_n t_n$ vanishes.

We maximise the *Lagrangian dual* with respect to $\boldsymbol \alpha$:

$$\begin{align}
& \underset{\alpha_1, \alpha_2,...,\alpha_N}{\max}\mathit{L}_\mathit{D}(\mathbf{a}) \quad = \quad  \sum_{n=1}^N \alpha_n - \quad \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^M \alpha_n \alpha_m t_n t_m \mathit{k}(\mathbf{x}_n , \mathbf{x}_m)\\ \\
& \text{subject to} \quad \sum_{n=1}^N \alpha_n t_n = 0, \quad \text{and} \quad  \alpha_n \geq 0 \quad \text{for} \ n = 1,...,N  
\end{align}$$

We replace the inner product $\boldsymbol{\phi}(\mathbf{x}_n)^T\boldsymbol{\phi}(\mathbf{x}_m)$ with the *kernel function* $\mathit{k}(\mathbf{x}, \mathbf{x'})$. The significance of this will become clear under our discussion of non-linear decision boundaries later on. Provisionally, it suffices to note that the substitution for the kernel function only occurs in the dual formulation of the Lagrangian, and not in the primal formulation. For now, it is best to think of it as (a function that jointly maps two arbitrary vectors onto) a scalar.

### Karush-Kuhn Tucker conditions

The Karush-Kuhn Tucker conditions are the following:

[WHAT ARE THEY conceptually - SOUNDBITES]

$$\begin{align}
\frac{\partial}{\partial w_m} \mathit{L}_\mathit{P} = w_m - \sum_{n=1}^N \alpha_n t_n \phi_m (\mathbf{x_n}) &= 0 \quad \text{for} \ m = 1,...M \\
\frac{\partial}{\partial b} \mathit{L}_\mathit{P} = - \sum_{n=1}^N \alpha_n t_n &= 0 \\ \\
t_n( \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 & \geq 0 \quad \text{for} \ n = 1,...N \\ \\
\alpha_n & \geq  0 \quad \forall n \\ \\
\alpha_n\left [t_n(\mathbf{w} ^T \boldsymbol{\phi}(\mathbf{x}_n) + b) -1\right ] &= 0 \quad \forall n
\end{align}$$

Burges: technical regularity assumption

The problem for SVMs is convex (a convex objective function, with constraints which give a convex feasible region), and for convex problems (if the technical regularity conditoon holds), the KKT are *necessary* and *sufficient* for $\mathbf{w}$, $b$, $\mathbf{a}$ to be a solution (Fletcher 1987)

Solving the SVM problem is equivalent to finding a solution to the KKT conditions above. 

A siginficant amount of work has been done to reformulate the problem into terms that are amenable to our tools.

### Support vectors

One of the conditions here, known as the *dual complementarity* condition is significant. It means that for every data point, either $\alpha_n = 0$, OR $t_n(\mathbf{w} ^T \boldsymbol{\phi}(\mathbf{x}_n) + b) = 1$.

The *support vectors* are those data points for which $\alpha_n > 0$, meaning that they also satisfy $t_n(\mathbf{w} ^T \boldsymbol{\phi}(\mathbf{x}_n) + b) = 1$. They correspond to the points that lie on either of the maximum margin hyperplanes in feature space.

The remaining points, those of which are *not* support vectors satisfy $\alpha_n = 0$, and also $t_n(\mathbf{w} ^T \boldsymbol{\phi}(\mathbf{x}_n) + b) > 1$.

Q - It seems from all expositions that both conditions cannot be satisfied simultaneously. Explain why. To do with whether or not constraint is active and positivity of Lagrange multipliers.

It must be emphasised that the support vectors are the *critical* elements of the training set; in the sense that if all other training points were removed, or moved around, but so as not to cross the maximum margin hyperplanes in feature space, and training was repeated, the same separating hyperplane would be found.

This is a quadratic programming problem, and it yields a solution in the form of numerical estimates $\hat\alpha_1, \hat\alpha_2,...\hat\alpha_N$.

We can then substitute to find an estimate $\mathbf{\hat w}$.

Bias parameter estimate determination, and numerical stability.

### Classfying new points

Once estimates $\hat\alpha_1, \hat\alpha_2,...\hat\alpha_N$, $\mathbf{\hat w}$, and $\hat b$ have been obtained, our training phase is complete. We classify new unlabelled test data points via the following:

$$\hat y( \mathbf{x}_{test}) = \mathbf{\hat w}^T \boldsymbol{\phi}(\mathbf{x}_{test}) + \hat b $$

We then evaluate the sign of the above to yield a classification:

$$ \hat f(\mathbf{x}_{test}) = sign [ \hat y(\mathbf{x}_{test} )] = sign[\mathbf{\hat w} ^T \boldsymbol{\phi}(\mathbf{x}_{test}) + \hat b] $$

We can also state the solution in terms of our estimates $\hat\alpha_1, \hat\alpha_2,...\hat\alpha_N$:

$$ \hat f(\mathbf{x}_{test}) = sign\left( \sum_{n=1}^N \hat \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_{test})^T \boldsymbol{\phi}(\mathbf{x}_n) + \hat b \right) = sign\left( \sum_{n=1}^N \hat \alpha_n t_n \mathit{k}(\mathbf{x}_{test}, \mathbf{x}_n) + \hat b \right) $$

In the second formulation, it becomes clear that only those points for which $\alpha_n > 0$ i.e. the *support vectors*, feature in the required calculations to classify a test point. As the remaining training points have $\alpha_n = 0$, they do not contribute to the predicted class label of a test point; and so predictions only directly depend on a subset of the training data. 

Note that we have made a similar substitution of the inner product $\boldsymbol{\phi}(\mathbf{x}_{test})^T \boldsymbol{\phi}(\mathbf{x}_n)$ with the kernel function $\mathit{k}(\mathbf{x}_{test} , \mathbf{x}_n)$, as in the Lagrangian dual formulation.

This has implications for computational complexity.

### Non-linear decision boundaries

The substitution of the inner product between feature mappings/basis functions for the kernel function is known as the *kernel trick*, and allows for extension of the hard-margin SVM to accommodate *non-linear decision boundaries* in input space. 

As this tutorial was written for the purposes of being self-contained, we will present cursory details on kernels as is necessary to illustrate their role in the extension of SVMs; and point in the direction of further references when necessary. 

The essence of the kernel trick in our context is that we do not need to explicitly specify the feature mapping/basis function transformations, but rather implicitly specify them through the kernel function. In our context, both the Lagrangian dual objective and the estimated decision function only feature the basis function vector as an inner product, and therefore this inner product can be substituted for a *suitable* choice of kernel function.

Furthermore, in the case where the data is not linearly separable in *input space*, it may be the case that the data turns out to be separable in *feature space*.

Frequently used functional forms for the kernel function in the SVM literature are:

- Linear                 : $\mathit{k}(\mathbf{x},\mathbf{x'}) = \mathbf{x}^T \mathbf{x'}$

- General Polynomial             : $\mathit{k}(\mathbf{x},\mathbf{x'}) = (r + \gamma \mathbf{x}^T\mathbf{x'})^p\ \quad r > 0$ 

- Radial basis           : $\mathit{k}(\mathbf{x},\mathbf{x'}) = \text{exp}\left( - \gamma \left \lVert \mathbf{x} - \mathbf{x'} \right \rVert_2^2 \right)$

- Sigmoid kernel :$\mathit{k}(\mathbf{x},\mathbf{x'}) = \text{tanh}\left (\gamma \mathbf{x}^T\mathbf{x'} + r \right)$

The introduction of these kernels introduces additional choices to be made for their hyperparameters. Examples are the $p$, $\gamma$ and $r$

To briefly illustrate the idea of defining a kernel, and how it implicitly defines a set of basis functions, an often used example is the polynomial kernel of degree $p = 2$:

For $\gamma = 1$ and $ r = 1$:

$$k(\mathbf{x}, \mathbf{x'}) = (1 + \mathbf{x} ^T \mathbf{x'} ) = (1 + x_1x'_1 + x_2x'_2)^2 \\ 
= 1 + 2x_1x'_1 + 2x_2x'_2 + (x_1x'_1)^2 + (x_2x'_2)^2 + 2x_1x'_1x_2x'_2 \\
=(1, \sqrt 2 x_1, \sqrt 2 x_2, x_1^2, \sqrt 2 x_1 x_2, x_2^2 ) (1, \sqrt 2 {x'}_1, \sqrt 2 {x'}_2, {x'}_1^2, \sqrt 2 {x'}_1 {x'}_2, {x'}_2^2)^T \\
=\boldsymbol{\phi}(\mathbf{x}) ^T \boldsymbol{\phi}(\mathbf{x'})$$

where $\boldsymbol{\phi}(\mathbf{x}) = (1, \sqrt 2 x_1, \sqrt 2 x_2, x_1^2, \sqrt 2 x_1 x_2, x_2^2 )^T$

This implicitly defines $M = 6$ basis functions,  $\phi_1(\mathbf{x}) = 1$, $\phi_2(\mathbf{x}) = \sqrt 2 x_1$, $\phi_3(\mathbf{x}) = \sqrt 2 x_2$, $\phi_4(\mathbf{x}) = x_1^2$, $\phi_5(\mathbf{x}) = x_2^2$, and $\phi_6(\mathbf{x}) = \sqrt 2 x_1 x_2$.

This polynomial kernel function represents an inner product in a 6 dimensional feature space. In this particular instance, we could also have defined the basis functions explicitly and that would amount to using the polynomial kernel. 

However, this is absolutely not a general rule. The kernel trick derives its usefulness from the fact that this in general does *not* hold. More concretely, the Gaussian form of the radial basis kernel function $k(\mathbf{x}, \mathbf{x'}) = \text{exp}\left(-\frac{1}{2 \sigma^2} \left \lVert \mathbf{x} - \mathbf{x'} \right \rVert^2_2 \right)$ uses an infinite number of basis functions. The feature space $\boldsymbol{\phi}(\mathbf{x})$ in this case is effectively infinite-dimensional.

It it therefore not actually possible to explicitly define the individual basis functions directly.

For the kernel trick to work, the kernel function must be a *Mercer kernel*. Necessary and sufficient conditions for the kernel function to be valid are that the Gram matrix $\mathbf{K}$, whose elements are given by $K_{nm} = k(\mathbf{x_n}, \mathbf{x_m})$, should be positive (semi-)definite.

Bishop (2006) and Hastie et al. (2013) acknowledge some contentions over a claim e.g. in Burges (1998) that kernelised SVMs were able to circumvent the curse of dimensionality. It seems that the consensus is that they do *not* circumvent the curse of dimensionality.

The argument Bishop (2006) gives is that the "coefficients weighting these different features are constrained to have different specific forms, and that any set of points in the original two-dimensional input space would be constrained to lie exactly on a two-dimensional nonlienar manifold embedded in the six-dimnensional feature space". However, it's not entirely clear to me exactly how that argument supports the conclusion.

Hastie et al. (2006) illustrate this on synthetic data sets in 12.3.4.

Whilst the essence of the kernel trick is conceptually simple to grasp, it invokes some theoretical results which we will not go into any depth on, but will list here. Some of the questions that we have failed to acknowledge, and which deeper theoretical results answer are the following, asked in Christmann and Steinwart (2008). 

- When is a function $k:X \ x \ X \rightarrow \mathbb{R}$ a kernel?
- How can we construct kernels?
- Given a kernel $k$, can we find a feature map and a feature space of $k$ in a constructive way?
- How does the kernel trick increase the expressive power of support vector machines?

The theory behind the answers to these questions invoke some results in functional analysis, and in reproducing kernel Hilbert Spaces (RKHSs). All of which require some mathematical groundwork, and which is the subject of another set of notes, forthcoming.

From a cursory literature review whilst these notes were being compiled, it turns out that the dividends of understanding the regularisation-RKHS viewpoint are:

- They enable the SVM to be cast in the more general framework of Tikhonov regularisation. This amounts to casting the SVM problem as a variational problem of finding a function that minimises a functional.

- They enable greater understanding of how it is that the variational infinite-dimensional problem can be recast into a finite dimensional optimisation problem, the finite dimensional solution of which is a solution to the regularisation problem. Hastie et al. (2011)

- Tikhonov regularisation can provide a broad framework from which to understand many ML regression and classification methods through a specification of a loss term and regulariser, which SVM can be compared to. 

- In particular, regularisation can be interpreted from a probabilistic, specifically, Bayesian perpsective as the introduction of priors, and under certain conditions on the priors, MAP estimation. 

- They enable us to perceive connections between regularisation theory and the statistical learning theory principles of *empirical risk minimisation*, *structural risk minimisation* and *capacity control*. Evgeniou et al. (2000).

### Soft-margin SVM

The *soft-margin SVM* amounts to a relaxation of the requirement for the data to be linearly separable in *feature space* $\boldsymbol{\phi}(\mathbf{x}_n)$. The resulting support vector machine will give exact separation of the training data in the original input space $\mathbf{x}$, but the decision boundary will be nonlinear.

The relaxation of the requirements amounts to an attempt to deal with training data which involves *overlapping class-conditional distributions*, and allows some of the training points to be misclassified, albeit incurring a penalty that increases as a linear function of the distance from the that boundary.

To formalise this, we introduce $N$ non-negative *slack variables* $\xi_n \geq 0 \quad n = 1,...N$ for each of the N training data points.

A data point $\mathbf{x}_k$ which is misclassified will have a class label $t_k$ of the opposite sign to that which is predicted $y(\mathbf{x}_k)$, and so their product $t_k y(\mathbf{x}_k)$ will be negative. This can only occur when $\xi_k > 1$.

For an arbitrary data point:

*Don't fully understand the cases and he logic - Excel value table

$ \xi_n > 1 $ - misclassified and lies on the wrong side of the decision boundary 

$ 0 < \xi_n \leq 1 $ correctly classified and lies on or inside the margin 

$ \xi_n = 0 $ correctly classified, lies 

$\xi_n = 1 $ correctly classified, lies on the decision boundary $y(\mathbf{x}_n) = 0$

[MOVE -The senstivity to outliers problem is not completely mitigated, as penalty for misclassification increases linearly with $\xi$. ]

The soft-margin SVM problem involves the following modifications to yield the following constrained optimisation problem:

$$\begin{align}
\underset{\mathbf{w}, b, \xi_1, \xi_2,...,\xi_n}{\min} \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + \mathit{C} \sum_{n=1}^N \xi_n \quad \text{subject to} \quad t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) &\geq 1 - \xi_n  \quad \text{for} \ n = 1,..., N \\ \xi_n & \geq 0 \quad \text{for} \ n = 1,...,N
\end{align}$$

Note that the objective function is modified to include the sum of $N$ slack variables $\sum_{n=1}^N \xi_n$, multiplied by a *regularisation hyperparameter*,  $\mathit{C}$.

 $\sum_{n=1}^N \xi_n$ -  an upper bound on the number of misclassified points.

 $\mathit{C} > 0$ - a regularisation hyperparameter that controls the trade-off between the slack variable penalty and the margin. 
 
 Alternatively, it controls the number of errors we are willing to tolerate on the training set (Murphy 2011). 
 
 A higher $C$ corresponds to a higher penalty for errors (Burges 1998). 
 
 As $\mathit{C}$ approaches $\infty$, this collapses into the linearly separable case.
 
As this is a hyperparameter, it can be tuned by using *cross-validation*, or by using an information criterion, options include the *Akaike Information Criterion* and the *Bayes Information Criterion*.

The $N$ hard-margin SVM constraints, $t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) \geq 1$ are replaced with the $N$ soft-margin SVM constraints $t_n(\mathbf{w} ^T\boldsymbol{\phi}(\mathbf{x}_n) + b) \geq 1 - \xi_n$

And are supplemented with a further $N$ constraints $\xi_n \geq 0$ due to the slack variables being non-negative.

Define $N$ Lagrange multipliers $\alpha_1, \alpha_2,..., \alpha_N$ corresponding to N inequality constraints, and stack them into an $N$-dimensional vector $ \mathbf{a} = (\alpha_1,\alpha_2,...,\alpha_N)^T$.

Define a further N Lagrange multipliers $\mu_1, \mu_2,...,\mu_N$ corresponding to the non-negativity constraints on the N slack variables $\xi_n$ and stack them into an $N$-dimensional vector $\boldsymbol{\mu} = (\mu_1, \mu_2,...\mu_N)^T$

Each Lagrange multiplier $\alpha_n, \xi_n \geq 0$

The *Lagrangian function* is therefore:

$$\mathit{L}(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2} \lVert \mathbf{w} \rVert^2_2 + \mathit{C} \sum_{n=1}^N \xi_n - \sum_{n=1}^N \alpha_n \left \{t_n y(\mathbf{x}_n) - 1 + \xi_n \right \} - \sum_{n=1}^N \mu_n \xi_n$$

We minimise the *Lagrangian primal function* with respect to $\mathbf{w}$, $b$ and $\boldsymbol{\xi}$:

$$\underset{\mathbf{w}, b, \boldsymbol{\xi}}{\min} \quad \mathit{L}_\mathit{P} (\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2} \lVert \mathbf{w} \rVert^2_2 + \mathit{C} \sum_{n=1}^N \xi_n - \sum_{n=1}^N \alpha_n \left \{t_n y(\mathbf{x}_n) - 1 + \xi_n \right \} - \sum_{n=1}^N \mu_n \xi_n$$

The corresponding *Karush-Kuhn Tucker conditions* are:

$$ \nabla_{\mathbf{w}} \mathit{L} = \mathbf{0} \implies \mathbf{w} = \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$$

$$\frac{\partial \mathit{L}}{\partial b} = 0 \implies \sum_{n=1}^N  \alpha_n t_n = 0 $$ 

$$\frac{\partial \mathit{L}}{\partial \xi_n} \implies \alpha_n = \mathit{C} - \mu_n $$

$$ \alpha_n \geq  0 \quad \forall n$$

$$ t_n( \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 \geq 0 \quad \text{for} \ n = 1,...N $$

$$\alpha_n\left \{t_n(\mathbf{w} ^T \boldsymbol{\phi}(\mathbf{x}_n) + b) -1\right \} = 0 \quad \forall n $$

$$ \mu_n \geq 0$$

$$\xi_n \geq 0$$

$$\mu_n \xi_n = 0$$

Similarly to the hard-margin Lagrangian dual, we can eliminate $\mathbf{w}$, $b$ and $\left \{\xi_n \right \}$, yielding the same Lagrangian dual objective as earlier , but with different constraints.

As $\alpha_n \geq 0$, $\mu_n \geq 0$, and $\alpha_n = \mathit{C} - \mu_n$, we can combine these constraints into what is known as *box constraints*, taking the form $ 0 \leq \alpha_n \leq \mathit{C}$

Recall that we *maximise* the *Lagrangian dual* with respect to each of the dual variables $\left \{\alpha_n \right\}$:

$$\begin{align}
& \underset{\alpha_1, \alpha_2,...,\alpha_N}{\max}\mathit{L}_\mathit{D}(\mathbf{a}) \quad = \quad  \sum_{n=1}^N \alpha_n - \quad \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^M \alpha_n \alpha_m t_n t_m \boldsymbol{\phi}(\mathbf{x}_n) ^T \boldsymbol{\phi}(\mathbf{x}_m)\\ \\
& \text{subject to} \quad \sum_{n=1}^N \alpha_n t_n = 0 \ , \ \quad \text{and} \quad  0 \leq \alpha_n \leq \mathit{C} \quad \text{for} \ n = 1,...,N  
\end{align}$$

[Can write this in more compact vector-matrix notation see scikit-learn.]

Classification is again the same as in the hard-margin SVM case using estimates $\hat\alpha_1, \hat\alpha_2,...\hat\alpha_N$, $\mathbf{\hat w}$, and $\hat b$

### Interpretation of the solutions

Using the box constraints $ 0 \leq \alpha_n \leq \mathit{C}$ to guide our interpretation of the solutions, we have three cases that we can elaborate on.

Case 1 - Data points with $\alpha_n = 0$, which do not contribute to the predictive model.

The support vectors are those vectors for which $\alpha_n > 0$, and which consequently satisfy the constraint $t_n y(\mathbf{x}_n) = 1 - \xi_n$,  yielding the following sub-cases:

Case 2 - Data points with $0 < \alpha_n < \mathit{C}$, which by virtue of constraint $\alpha_n = \mathit{C} - \mu_n $ imply that $\mu_n > 0$ and via constraint $\mu_n \xi_n = 0$ imply that $\xi_n = 0$.

These points lie *on* the margin (check against above classification and tidy up)

Case 3 - Data points with $\alpha_n = \mathit{C}$, by virtue of constraint $\alpha_n = \mathit{C} - \mu_n$, imply that $\mu_n = 0$. 

These points lie *inside* the margin

What information/restrictions on $\xi_n$ does constraint $\mu_n \xi_n = 0$ then yield? My reasoning is that it can be $\xi_n = 0$ or $\xi_n > 0$. 

Bishop's exposition: In this case, the data points can lie inside the margin and can either be correctly classified if $\xi_n \leq 1$ or misclassified if $\xi_n > 1$.

Hastie's exposition: the remainder of data points ($\hat \xi_n > 0$) have $\hat \alpha_n = \mathit{C}$. From 12.14 any of these margin points ($ 0 < \hat \alpha_n , \hat \xi_n = 0$) can be used to solve for the bias parameter.

Ah okay - both expositions are consistent with one another.

Note that only those case 2 support vectors with $0 < \alpha_n < \mathit{C}$, $\xi_n = 0$ so that $\t_n y(\mathbf{x}_n = 1$ are used in the determination of the bias parameter. 

[This can be more comprehensive in terms of the analysis of the constraints]

Determination of bias parameter and numerical stability - come back to this.

### SVM as loss/error minimisation

The *maximum-margin hard SVM* can be expressed in terms of the minimisation of an error function with a quadratic regularisation term:

$$\underset{\mathbf{w}, b}{\min}\sum_{n=1}^N \mathit{E}_{\infty}(y(\mathit{x}_n)t_n - 1) + \lambda \lVert \mathbf{w} \rVert_2^2 \quad \quad \mathit{E}_{\infty}(z) = \left\{
 \begin{array}{ll}
      0 & \quad \text{if} \ z \geq 0  \\
      \infty & \quad \text{otherwise}
 \end{array}
\right.
$$

The regularisation parameter $\lambda$ has only qualitative signficance here, and as long as $\lambda > 0$, its quantitative value plays no role.

The *maximum-margin soft SVM* can be expressed in terms of the minimisation of the *hinge loss/error function* with quadratic regularisation, taking the form:

$$\underset{\mathbf{w}, b}{\min}\sum_{n=1}^N \mathit{E}_{SV}(y_n , t_n) + \lambda \lVert \mathbf{w} \rVert_2^2 \quad \quad \mathit{E}_{SV}(y_n , t_n) = \left [1 - t_n y_n \right ]_{+} = {\max}(0, 1 - t_n y_n)
$$

where $\lambda = (2\mathit{C})^{-1}$, $\mathit{E}_{SV}(\cdot)$ is the *hinge loss/error function*, and where $[\cdot]_+$ denotes the positive part.

The hinge loss function is can be interpreted as an approximation to misclassification error. It is *convex*, but *not* differentiable due to the presence of the max term.

### $\boldsymbol{\nu}$-SVM

An alternative, equivalent formulation of the support vector machine, the $\nu$-SVM, involves the following maximisation problem:

$$\begin{align}
& \underset{\alpha_1, \alpha_2,...,\alpha_N}{\max}\mathit{L}_\mathit{D}(\mathbf{a}) \quad = \quad - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^M \alpha_n \alpha_m t_n t_m \boldsymbol{\phi}(\mathbf{x}_n) ^T \boldsymbol{\phi}(\mathbf{x}_m)\\ \\
& \text{subject to} \quad \sum_{n=1}^N \alpha_n t_n = 0 \, \quad \sum_{n=1}^N \alpha_n \geq \nu , \ \quad \text{and} \quad  0 \leq \alpha_n \leq \frac{1}{N} \quad \text{for} \ n = 1,...,N  
\end{align}$$

The hyperparameter $\nu$, which replaces the hyperparameter $\mathit{C}$ in the soft-margin SVM.

It can be tuned to take values in the range $0 < \nu \leq1$, and methods for tuning are the same as for $\mathit{C}$.

It can be interpreted as both an upper bound on the fraction of *margin errors* (points for which $\xi_n > 0$ and hence which lie on the wrong side of the *margin boundary* and which may or may not be misclassified) and a lower bound on the fraction of support vectors.

We can relate the hyperparameter $\nu$, the *fraction of misclassified points* with $C$, the *number* of misclassified points we are willing to tolerate on a training set, and common practice is to define $\mathit{C} = 1 \ / \ \nu \mathit{N}$.

$\nu$-SVM can then more precisely be described as a *reparametrisation* of the soft-margin SVM. 

### Numerical solutions for the quadratic programming problem and computational complexity considerations:

During the training phase (i.e. the determination of estimates of the parameters and the bias parameter), the whole data set is used, requiring that efficient algorithms for solving the quadratic programming problem.

Direct solution of quadratic programming problem via traditional techniques (?) is often feasible due to computational and memory requirements, require clever practical approaches.

Here are some of the techniques offered in Bishop (2006): 

*Chunking* (Vapnik 1982) exploits the invariance of value of the Lagrangian if we remove rows and columns of the *kernel matrix* corresponding to Lagrange multipliers with value zero. The full quadratic programming problem can be broken down into a series of smaller problems, whose goal is to identify all the nonzero Lagrange multipliers and discard the others.

*Chunking* can be implemented using *protected conjugate gradients* (Burges, 1998). This reduces the size of the matrix in the quadratic function to approximately the number of nonzero Lagrange multipliers squared, this may be too demanding memory-wise for large-scale applications.

A very well-known and one of the most popular approaches is *sequential minimal optimisation* or *SMO* (Platt, 1999). it takes chunking to an extreme and considers two Lagrange multipliers at a time, yielding analytically soluble subproblems without recourse to numerical quadratic programming methods. Heuristics are given for choosing the pair of Lagrange multipliers at each step. In practice, SMO is found to ahve scaling with the number of data points somewhere between linear and quadratic depending on the application.

*Decomposition methods* (Osuna et al., 1996) also solve a series of smaller quadratic programming problems, and these are designed to so that each of these are a fixed size, so the techniques can be applied to arbitrarily large data sets. This still involves numerical solution of quadratic orogramming subproblems - problematic and expensive.

### Probabilistic extensions

Crucially, the *support vector machine* is known in the literature as a *non-probabilistic classifier* - it makes classification decisions for new input vectors, and does not provide any outputs that can be *formally* interpreted as probabilistic.

However, in a very loose sense, the output of the decision function $\hat y$ can be interpreted as an uncalibrated "confidence" value. Many techniques were suggested in the late 90s for calibrating the SVM output into a probability, a brief survey of which can be found in Platt (2000).

In the same paper, a notable technique was proposed by Platt (2000) to fit logistic sigmoids to the outputs of a previously trained support vector machine by setting the conditional probability to be of the form:

$$p(t = 1 | \mathbf{x}) = \sigma \left( A y(\mathbf{x}) + B \right) $$

where $y(\mathbf{x})$ is defined as before. $A$ and $B$ are estimated by maximum likelihood by minimising a cross-entropy error function defined by a separate training set to avoid severe over-fitting. This is equivalent to interpreting the output of the decision function $\hat y$ as the log-odds ratio of $\mathbf{x}$ belonging to class $t = 1$ i.e. $\text{ln} \frac{p(t = 1 | \mathbf{x})}{p(t = -1 | \mathbf{x})}$

It was subsequently observed by Tipping (2001) that this method can give poor approximations to the posterior probabilities. This motivated development of *relevance vector machines*, a Bayesian characterisation of SVMs which were designed to overcome the issue of SVMS being non-probabilistic, as well as a number of other drawbacks. More details can be found in Tipping (2001) and Bishop (2006).

### Properties of the solutions

A key result from convex optimisation is that given a convex objective function and given the constraints define a convex region, any local optimum will also be a global optimum.

As the constraints are linear in $\mathbf{w}$?, the constraints fulfil the convexity criteria. 

*Need to qualify precisely concerning the objective as Bishop is a little unclear here.

### Multi-class classification

The support vector machine is by design a *two-class*, binary classifier, but there are many occasions where we need to classify $K > 2$ classes. A number of ad-hoc extensions have been proposed, and we give a survey of these methods from Bishop (2006).

*One-versus-the-rest* - this was suggested by (Vapnik 1998), and involves the construction of $K$ separate SVMs in which the $k$th model $y_k(\mathbf{x})$ is trained using the data from class $C_k$ as the positive examples, and the data from the remaining $K-1$ classes as the negative examples. This amounts to solving $K$ QP optimisation problems. However, it suffers from the problem that the decisions of the individual classifers can lead to inconsistent results in which an input is assigned to multiple classes simultaneously.  The following visualisation illustrates the problem well:

A solution can be addressed by making predictions for new inputs $\mathbf{x}$ using:

$$y(\mathbf{x}) = \underset{k}{\max} y_k(\mathbf{x})$$

However, this heuristic approach suffers from a further problem - the different classifiers were trained on different tasks, and there is no guarantee that the real-valued quantities $y_k(\mathbf{x})$ for different classifiers will have appropriate scales.

A further problem with the *one-versus-rest* approach is that the training sets are *imbalanced*. As an example, for $K = 10$ classes each with equal numbers of training data points, then the individual classifers are trained on data sets comprising 90% negative examples and only 10% positive examples, and the symmetry of the original problem is lost. Variants on this were proposed by Lee et al. (2001), who modify the target values so that the positive class has target value $+1$ and the negative class has target $-1/(K-1)$.

Training all $K$ *SVMS simultaneously* with a single objective function by maximising the margin from each to remaining classes was suggested by Weston and Watkins (1999). The main drawback is the computational cost - solving $K$ optimisation problems over $N$ data points yields and overall cost of $O(KN^2)$, in this case, a single optimisation problme of size $(K-1)/N$ must be solved with overall cost of $O(K^2N^2)$. A variant on this is given in Crammer and Singer (2001)

*One versus one* - train $K(K-1)/2$ different 2-class SVMs on *all possible pairs* of classes, and then classify test points according to which class has the highest number of votes (i.e. majority voting). It originated in context of neural networks (Knerr et al. 1990 as cited in Hsu and Lin (2002)), and later applied to SVMs (Kressel et al. 1999 as cited in Hsu and Lin (2002) . Similarly has the issue of how to resolve ties between predicted classifications as with *one-versus-rest*. For large $K$ this incurs computational penalties during training *and* testing of a greater degree than the *one-versus-rest* approach.

*DAGSVM* - organises pairwise classifiers into a *directed acyclic graph* (Platt 2000). For $K$ classes, the DAGSVM has a total of $K(K-1)/2$ classifiers, and classifying a new test points requires only $K-1$ pairwise classifiers to be evaluated, with classifiers used depending on the path through the graph traversed. This 

*Error-correcting output codes* - an information theoretic/coding theory approach developed by Dietterich and Bakiri (1995) for use with binary concept learning algorithms such as C4.5 decision trees, and applied to support vector machines by Allwein et al. (2000). The K classes themselves are represented as particular sets of responses from the two-class classifiers chosen, together with a suitable decoding scheme, that gives robustness to errors and to ambiguity in the outputs of the individual classifiers.

Whilst these approaches are illuminative, we cannot shake off the feeling that they are *ad-hoc, heuristic* engineering extensions. Bishop's opinion (2006) is that the application of SVMs to multiclass classification is an open issue, and that in practice, the *one-versus-rest* approach is most widely used in spite of this ad-hoc flavour and the practical limitations. Murphy's opinion (2012) is that these issues fundamentally arise due to SVMs not modelling uncertainy and outputting probability scores, leading to lack of comparison across classes. Due to the explicit focus of Bishop's text on Bayesian formulations, and also Murphy's explicit focus on probabilistic algorithms (of which Bayesian formulations can be viewed as a subset); it is not difficult to understand some of their objections given their theoretical predispositions.

### Single-class classification

The *single-class* support vector machine is somewhat related to the unsupervised learning problem of *probability density estimation*. Thse methods do not model the density of the data, but aim to find a smooth boundary enclosing a region of high-density. The boundary is chosen to represent a quantile of the density, that is the probability that a data point drawn from the distribution will land inside that regios is given by a fixed number between 0 and 1 that is specified in advance. In ore formal terms, this amounts to estimating the *support* of a (possibly high-dimensional) probability distribution.

Two approaches to this problem are by Scholkopf et al. (2001), who try to find a hyperplane that separates all but a fixed fraction $\nu$ of the training data from the origin whilst at the same time maximising the margin of the hyperplane from the origin. Tax and Duin (1999) look for the smallest sphere in feature space that contains all but a fraction $\nu$ of the data points. For kernels $k(\mathbf{x},\mathbf{x}')$ that are functions only of $\mathbf{x} - \mathbf{x}'$, the two algorithms are equivalent. 

### Implementation in scikit-learn

The *scikit-learn* Python package implements SVMs using *LIBSVM* and *LIBLINEAR*, details of which can be found in Chuang and Lin (2001) and Lin et al. (2008). We advise that the SVM scikit-learn user guide is consulted, and only offer some high level comments on the implementations, by making these comments in context of the methods we have presented in these notes. 

LIBSVM is the default implementation, and LIBLINEAR is designed primarily with a view to handling large datasets (but probably not at commercial production-grade level) and also for greater flexibility in specification of penalties and loss functions.

- *Multiclass classification* in LIBSVM is implemented using the *one-vs-one* approach. In LIBLINEAR, it is implemented using the *one-vs-rest* and an alternative *multiclass* strategy in the spirit of Crammer and Singer (2001).

- *Calibrated probability outputs* from scores in LIBSVM are implemented via the Platt scaling method, described in the above sections and contained in the paper Platt (1999). For multiclass classification, this is implemented via the work of Wu et al. (2004).

- *Kernels* in LIBSVM are also accomodated, with the default options being the ones listed above, i.e. linear, polynomial, RBF, and sigmoid.

- *Custom kernels* in LIBSVM are implemented by specifying a kernel function, or the Gram matrix.

- *Single-class classification* is implemented in LIBSVM using the work of Scholkopf et al. (2001)

- *C* hyperparmater and gamma in RBFs

### Relations to other techniques

- Prior work on perceptrons (Rosenblatt, 1958). Illustrate the provision of an optimality criterion for choosing between competing hyperplanes in the form of maximum margin criteria, together with relevant discussion of the perceptron's drawbacks.

- Margins as an explanation of AdaBoost 

- Relation to logistic regression (~), hinge loss properties and other loss functions.

- What is characteristic of Bishop vs Hastie vs Murphy? Distil

- Areas that need more work, optional or can acknowledge

### A generalisation bound

When we initially motivated our tutorial, we said that the essence of the support vector machine is that it supplies an optimality criterion for choosing from an (infinite) number of separating hyperplanes - the optimal classifier is that which maximises the margin.

Whilst the optimality criterion of finding an optimal separating hyperplane that maximises the margin for *training data* (as shown in our visualisation) is intuitively plausible, we have backgrounded the question of whether the optimality criterion of maximising the margin is supported by any theoretical considerations, and how it relates to the generalisation performance of the classifier/classification rule it induces.

From the original paper of Boser, Guyon, and Vapnik (1992), with our emphasis in italics added:

"Good generalization performance of pattern classifiers is achieved when the *capacity* of the classification function is matched to the size of the training set. Classifiers with a large number of adjustable parameters and therefore large capacity likely learn the training set without error, but exhibit poor generalisation. Conversely, a classifier with insufficient capacity might not be able to learn the task at all. In between there is an optimal capacity of the classifier which minimises the *expected generalisation error* for a given amount of training data. Both experimental evidence and theoretical studies link the generalisation of a classifer to the error on the training examples and the complexity of the classifier."

*Brief comment on Burges analogy, SVM development context, general ML bias-variance tenet applied to squared loss in regression setting, provenance of bias-variance in ML, and in statistics.

There is a significant amount of theory that underlies the relation between the capacity of a learning machine (function estimation algorithm) and its generalisation performance (generalisation accuracy or error rate as test set size tends to infinity). It grew out of considerations of under what circumstances, and how quickly, the mean of some empirical quantity converges uniformly, as the number of data points increases, to the true mean (that which would be calculated from an infinite amount of data) (Vapnik, 1979 as cited in Burges, 1998).

We will present only enough results to better understand some of the theoretical motivations underlying the development of SVMs. Significantly, we invoke some results from statistical learning theory that are times illustrative, and crucially, we emphasise the limits of the insights that this theory can provide. Much of this exposition is from the excellent tutorial by Burges (1998).

The notation in this section will follow Vapnik (1995):

Suppose we have $l$ observations, with each $i$ th observation consisting of an input-vector, output-label pair $(\mathbf{x_i}, y_i)$, where $\mathbf{x_i} \in \mathbb{R}^N$ and $y_i \in \{+1, -1\}$. 

Assume there exists an *unknown* joint probability distribution $P(\mathbf{x}, y)$ from which the data (i.e. input-vecotrs) is generated *iid (indepedently and identically distrbuted)*. Here $P$ stands for a cumulative probability distribution and $p$ for the density. Additionally, there is the nuance that this assumption is more general than associating a fixed $y$ with every $x$; it allows for a distribution of $y$ for a given $x$. In that case, the "teacher" would assign labels $y_i$ according to a fixed distribution, conditional on $\mathbf{x_i}$.

We would like our function estimation algorithm to estimate a mapping $\mathbf{x}_i \mapsto y_i$. The algorithm is actually defined by a set of possible mappings $\mathbf{x} \mapsto f(\mathbf{x}, \alpha)$, where the functions $f(\mathbf{x}, \alpha)$ are parametrised by $\alpha$. The function estimation algorithm is deterministic, that is for a given input $\mathbf{x}$, and a choice of $\alpha$, it will always give the same ouput $f(\mathbf{x}, \alpha)$. A function estimation algorithm with a particular choice of parametrisation $\alpha = \alpha_0$ is a "trained macihine".

The *expectation of the test error* for a "trained machine", or *expected risk* is defined as:

$$R(\alpha) = \int \frac{1}{2} \left \lvert y - f(\mathbf{x}, \alpha) \right \rvert dP(\mathbf{x},y)$$ 

Cosmetically, when a density $p(\mathbf{x}, y)$ exists, we can rewrite $dP(\mathbf{x}, y)$ can be rewritten $p(\mathbf{x}, y) d\mathbf{x} dy$. However, that does not change the fact that the distribution $P(\mathbf{x},y)$ is unknown, unless we have an estiamte of what it is. 

The *empirical risk* $R_{emp}(\alpha)$ is defined as the *measured mean error rate on the training set for a fixed finite number of observations*:

$$R_{emp}(\alpha) = \frac{1}{2l} \sum_{i=1}^l \left \lvert y_i - f(\mathbf{x}_i, \alpha) \right \rvert$$

No probability distribution appears here. $R_{emp}(a)$ is a fixed number for a particular choice of parameter $\alpha$ and for a particular training set $\left \{\mathbf{x}_i, y_i \right \}$. The quantity $\frac{1}{2} \lvert y_i - f(\mathbf{x}_i, \alpha) \rvert$ is called the *loss*, and it can only take values between 0 or 1. This specific functional form is referred to in the literature as *zero-one loss*.

For some $\eta \in [0, 1]$, for losses taking these values, with probability $1 - \eta$, the following bounds holds (Vapnik 1995 as cited in Burges 1998):

$$R(\alpha) \leq R_{emp}(\alpha) + \underbrace{\sqrt{\left(\frac{h(\text{log}(2l/h) + 1) - \text{log}(\eta/4)}{l} \right)}}_{\text{VC confidence}}$$

where $h$ is a non-negative integer called the *Vapnik-Chervonenkis (VC) dimension*, measuring the notion of *capacity*. The entirety of the right hand side is referred to by Burges as the *risk bound*, and the second term on right hand side is referred to in the statistical learning literature as the *VC confidence*.

Three observations:

1. The bound is indepedent of $P(\mathbf{x}, y)$ and only assumes that both the training and test data are drawn independently according to *some* $P(\mathbf{x}, y)$.

2. It is not in general possible to compute the left hand side, i.e. the expected risk $R(\alpha)$.

3. If we know *h*, we can compute the right hand side. Given different "learning machines" (a learning machine is a *family of funcitons* $f(\mathbf{x}, \alpha)$), and choosing a fixed, sufficiently small $\eta$, by choosing that machine which minimises the right hand side, we are choosing that machien which gives the *lowest upper bound* on the *expected risk *.

Observation 3. yields a principled method for choosing a learning machine given a task, and is the essential idea behind *structural risk minimisation*. Given a fixed family of learning machines to choose from, to the extent that the bounds is tight for at least one of the machines, one will not be able to do better than this. To the extent that the bound is not tight for any, the hope is that the right hand side still gives useful information as to which learning machine minimises the expected risk.

Q: What do tight bounds mean?

Q: Restrictions on the family of functions

The VC dimension is a property of a set of functions $\{f(\alpha)\}$ (again using $\alpha$ as a generic set of parameters - a choice of $\alpha$ parametrises a specific function), and can be defined for various classes of function $f$. We will only consider functions that correspond to the two-class pattern recognition case, so that $f(\mathbf{x}, a) \in \{+1, -1\} \forall \mathbf{x}, \alpha$.

Now if a given set of $l$ points can be labelled in all possible $2^l$ ways, and for each particular labelling, a member of the set $\{f(\alpha)\}$ can be found which correctly assigns those labels, we say that the set of points is *shattered* by that set of functions.

The *VC dimension* for the set of functions $\{f(\alpha)\}$ is defined as the *maximum number of training points* that can be shattered by $\{f(\alpha)\}$. 

If the VC dimension is $h$ then there exists *at least one set* of $h$ points that can be shattered, but in general, it will not be true that *every* set of $h$ points can be shattered. 

Shattering points with oriented hyperplanes in $\mathbb{R}^N$

As the definition is quite general, here is an illustration of the principle. Suppose that the sapce in which the data live is $\mathbb{R}^N$, and the set $\{f(\alpha)\}$ consists of oriented straight lines, so that for a given line, all points lie on one side are assigned the class 1, and all points on the other side, the class -1. The orientation is shown in Figure 1 by an arrow, specifying on which side of the line points are to be assigned hte label 1. While it is not possible to find three points that can be shattered by this set of functions, it is not possible to find four. So the VC dimension of the set of oriented lines in $\mathbb{R}^2$ is three.

Generalising to hyperplanes in $\mathbb{R}^N$, the following theorem:

Consider some set of $m$ points in $\mathbb{R}^N$. Choose any one of the points as origin. Then the m points can be shattered by oriented hyperplanes if and only if the position vectors of the remaining points are *linearly independent*.

Corollary: The VC dimension of the set of oriented hyperplanes in $\mathbb{R}^N$ is $n+1$, since we can always choose $n+1$ points, and then choose one of the points as origin, such that the position vectors of the remaining *n* points are linearly indepednety, but can never choose $n+2$ such points, (as no $n+1$ vectors in $\mathbb{n}$ can be linearly independent.

VC dimensions and the number of parameters

The VC dimension gives a formalised definition of the notion of the capacity of a given set of functions (or hypothesis class). Intuitively, one might expect that the learning machines with many parameters would have high VC dimension, while learning with few parameters would have low VC dimension.

However, there are some caveats. A notable counterexample of a learning machine with just one parameter but infinite VC dimension was proposed by Levin and Denker, in Vapnik and cited in Burges. A family of classifiers is said to have infinite VC dimension if it can shatter $l$ points, no matter how large $l$.

Furthermore, even though we can shatter an arbitrarily large number of points (infinite VC dimension), the nuance here is that the VC dimension refers to the *maximum* number of training points that can be shattered; and in the above case, we can also find a situation where just four points cannot be shattered.

Minimising the bound by minimising the VC dimension.

![Fig.%203.png](attachment:Fig.%203.png)

The above figure shows how the VC confidence term varies with the VC dimension $h$, given a 95% confidence interval, correspoonding to $\eta = 0.05$, and assuming a training sample size of 10,000, corresponding to $l = 10,000$.

The VC confidence is a *monotonic, increasing* function of $h$, and is true for *any* value of $l$, meaning that this result is independent of the size of the training set.

Given some selection of learning machines whose empirical risk zero, one wants to choose that learning machine whose associated set of functions has *minimal VC dimension*, yielding a better upper bound on the expected error. In general, for non-zero empirical risk, one wants to choose that learning machine which minimises the entire right hand side of Eq. (3).

A caveat on interpreting Eq. (3) is that its a probabilistic upper bound on the expected risk. This does not prevent a particular machine with the same value for empirical risk, abd whose function set has higher VC dimension, from having better performance. A good example of this the kth nearest neighbour classifier with $k = 1$ - it has zero empirical risk and infinite VC dimension. Any number points labelled arbitrarily will be succesfully learned by the algorithm, and thus the bound provides no information. In fact, for any classifier with infinite VC dimension, the bound is not even valid. However, even thoguh the bound is not valid, nearest neighbour classifiers can still perform.

The key cautionary tale here is that infinite capacity does *not* guarantee poor performance.

Structural risk minimisation

With these tools, we can summarise principle behind *structural risk minimisation (SRM)*, (Vapnik 1979 as cited in Burges 1998).

The VC confidence term depends on the chosen *class of functions*, whereas the empirical and expected risk depend on the one particular funciton chosen by the training procedure. We would like to find that subset of the chosen set of functions, such ta thte risk bound for that subset is minimised. As we cannot vary the integer-valued VC dimension smoothly, we have to introduce a "structure" by dividing the entire class of functions into nested subsets. For each subset, we must be able to either compute $h$ or to get a bound on $h$ itself. 

SRM then consists of finding that subset of functions which minimises the bound on the expected risk. This can be done by simply training a series of machines, one for each susbset, where ffor a given subset the goal of trianing is simply to minimise the empirical risk. One then takes that trained machine in the series whose sum of empirical risk and VC confidence is minimal.

### VC dimension of SVMs

The VC dimension of SVMs can be very large and even infinite. However, SVMs still exhibit good generalisation performance. There exist some plausibility arguments as to why, but there a rigorous theory which *guarantees* that a given family fo SVMS will ahve a high accuracy on a given problem is still an open question.

We call any kernel that satisfies Mercer's condition a *positive kernel*, and the corresponding space $\mathcal{H}$ the *embedding space*. We will call any embeddding space with *minimal dimension* for a given kernel a *minimal embedding space*.

The following theorem holds:

Let $K$ be a positive kernel which corresponds to a minimal embedding space $\mathcal{H}$. Then the VC dimension of the corresponding support vecotr machine (where the regularisation hyperparameter/error penalty $C$ can take all values) is $dim(\mathcal{H}) + 1$. 

And:

If the space in which the data live has dimension $d_L$ (i.e. $\mathcal{L} = \mathbb{R}^{d_L}$), the dimension of hte minimal embedding space for homogeneous polynomial kernels of degree $p$ $ \left (K(\mathbf{x}_1, \mathbf{x}_2) = \left (\mathbf{x}_1 ^T \mathbf{x}_2\right )^p, \mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^{d_L} \right)$ is $\binom{d_L + p + 1}{p} + 1$

And:

Consider the class of Mercer kernels for which $K(\mathbf{x}_1, \mathbf{x}_2) \rightarrow 0$ as $\left \lVert \mathbf{x}_1 - \mathbf{x}_2 \right \rVert_2 \rightarrow \infty$ and for which $ K(\mathbf{x}, \mathbf{x}) $ is $O(1)$, and assume that the data can be chosen arbitrarily from $\mathbb{R}^d$. Then the family of classifiers consisting of support vector machines using these kernels, and for which the error penalty is allowed to take all values, has *infinite VC dimension*. 

Additional discussion in Burges

There is still the conundrum of how it is that even though the VC dimension is infinite (if the data is allowed to take all values in $R^{d_L}$), SVM RBFs can have excellent performance (Scholkopf et al. 1997 as cited in Burges 1998). And similarly for polynomial SVMs. Why?

### Generalisation performance of SVMs - issues

Can we explain the generalisation performance of SVMs entirely through structural risk minimisation (SRM) principles? Here is an excellent discussion by Burges (1998), on whether SVMs *rigorously implement* structural risk minimisation.

Significantly, it turns out that the original argument for structural risk minimisation for SVMs is known to be flawed, since the structure there is determined by the data (Section 5.12, Vapnik 1999, as cited in Burges 1998). In particular, the original monograph (Vapnik 1999) states that to define the structure on the set of linear functions we use the set of canonical hyperplanes constructed with respect to vectors $x$ from the training data, whilst *according to the SRM principle, the structure has to be defined a priori before the training data appear*. 

However, it is known how to do structrual risk minimisation for systems where the structure does depend on the data (Shawe Taylor et al., 1996a; Shawe-Taylor et al., 1996b as cited in Burges 1998). Unfortunately the bounds are much looser than the VC bounds, which are already very loose.

Furthermore, structural risk minimisation can be rigorously implemented without the above violation for a very similar class of classifiers known as *gap-tolerant classifiers*. The signature feature these classifiers involve is that of a minimal closing sphere of a particular diameter $D$. Considerations of the VC dimension of these classifiers together with some more obscure bounds suggest that algorithms that minimise $D^2/M^2$ can be expected to give better generalisation performance (Burges 1998).

### Historical development and areas for further reading

Pedagogically, it is important to place a technique into a wider historical and theoretical context. Some modern presentations (such as in textbooks) often sever the technique from its context, together with the motivations of its progenitors. We believe that better appreciating the historical, path-dependent nature of how a technique evolves can reinforce a technique's significance.

-SVMs introduced in the paper by Boser, Guyon, Vapnik (1992). The kernel trick was accredited to Aizerman.

### Footnotes

### References

https://en.wikipedia.org/wiki/Support-vector_machine

https://scikit-learn.org/stable/modules/svm.html#classification

Xing, E., Lecture 6, Support vector machines, Carnegie Mellon University, 10-701 Machine Learning Fall 2016

Paisley, J. Lecture 11, Maximum margin classifier, Columbia University, ColumbiaX: Machine Learning 

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Ch. 7 – Sparse kernel machines.

Hastie, T., Tibshirani, R., Friedman, J. (2009).  The Elements of Statistical Learning Ch. 12 – Support vector machines and flexible discriminants.

Murphy, K. (2011). Machine Learning: A Probabilistic Perspective. Ch. 14.5 Support vector machines (SVMs)

Burges, C. (1998). A tutorial on support vector machines for pattern recognition

Vapnik, V. (1999). An overview of statistical learning theory

Boser, B., Guyon, I., Vapnik, V. (1992). A training algorithm for optimal margin classifiers.

Platt, J. (2001). Fast training of support vector machines using sequential minimal optimisation.

Platt, J. (1999) . Probabilistic outputs for support vector machines and comparisons to regularised likelihood models. 

Scholkopf, B., Smola A. (2002). Learning with Kernels - Support Vector Machines, Regularization, Optimization and Beyond

Fan, R., Chang F, Hsieh C., Wang X., Lin C., (2008). LIBLINEAR: A library for large linear classification.

Chang C., Lin C. (2001). LIBSVM - A library for support vector machines.