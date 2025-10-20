# Evaluating Generative Models: From KL Divergence to Sample-Based Metrics

This document explains two key concepts in generative modeling:
1.  **Kullback-Leibler (KL) Divergence** as applied to mixture distributions, focusing on the difference between Forward and Reverse KL.
2.  **Sample-Based Evaluation** using modern metrics like Sliced Wasserstein Distance and Fréchet Distance.

---

## Part 1: KL Divergence with Mixtures

### What is KL Divergence?

The **Kullback-Leibler (KL) Divergence** is a measure of how one probability distribution, $P$, is different from a second, reference probability distribution, $Q$. It's an asymmetric measure, meaning $D_{KL}(P || Q) \neq D_{KL}(Q || P)$.

The general formula is:
$$
D_{KL}(P || Q) = \int p(x) \log\left(\frac{p(x)}{q(x)}\right) dx = \mathbb{E}_{x \sim P} \left[ \log(p(x)) - \log(q(x)) \right]
$$

**The Challenge with Mixtures:**
When $P$ or $Q$ are mixture models (like Gaussian Mixture Models, GMMs), this integral becomes intractable. For example, if $p(x) = \sum \pi_i p_i(x)$ and $q(x) = \sum \omega_j q_j(x)$, the $\log$ of a sum prevents a closed-form solution.

$$
D_{KL}(P || Q) = \int \left(\sum_i \pi_i p_i(x)\right) \log\left(\frac{\sum_i \pi_i p_i(x)}{\sum_j \omega_j q_j(x)}\right) dx
$$

This integral is hard to compute. However, the *choice* of which distribution is $P$ (the data) and which is $Q$ (the model) leads to two different optimization objectives with drastically different behaviors.

### Forward KL: $D_{KL}(P || Q)$ (Mode-Covering)

In this setup, we let $P$ be the true data distribution and $Q$ be our generative model. We want to minimize $D_{KL}(P || Q)$.

* **Formula:** $D_{KL}(P || Q) = \mathbb{E}_{x \sim P} [-\log q(x)] + \text{const.}$ (This is the objective in standard Maximum Likelihood Estimation, MLE).
* **Interpretation:** We average over *real data samples* ($x \sim P$).
* **Penalty:** The objective $D_{KL}(P || Q) \to \infty$ if $q(x) \to 0$ for any $x$ where $p(x) > 0$.
* **Behavior (Mode-Covering):** To avoid an infinite penalty, the model $Q$ *must* place probability mass everywhere the real data $P$ has mass. If $P$ has multiple modes (e.g., a GMM with two peaks), $Q$ is forced to "cover" both. To do this efficiently, $Q$ will often become a single, broad distribution that covers all the modes of $P$, placing mass in the low-probability regions *between* the modes.
* **Result:** Often leads to "blurry" or "averaged" results. For example, a Variational Autoencoder (VAE) trained to minimize this objective might generate a fuzzy average of a "3" and an "8" if those are two modes in the data.



### Reverse KL: $D_{KL}(Q || P)$ (Mode-Seeking)

In this setup, we let $Q$ be our generative model and $P$ be the true data distribution. We want to minimize $D_{KL}(Q || P)$.

* **Formula:** $D_{KL}(Q || P) = \mathbb{E}_{x \sim Q} [\log q(x) - \log p(x)]$
* **Interpretation:** We average over *generated samples* ($x \sim Q$).
* **Penalty:** The objective $D_{KL}(Q || P) \to \infty$ if $q(x) > 0$ for any $x$ where $p(x) \to 0$.
* **Behavior (Mode-Seeking):** To avoid this penalty, the model $Q$ must *not* place probability mass where the real data $P$ has no mass. It is "safer" for $Q$ to pick just *one* mode of $P$ and model it perfectly (where $p(x)$ is high) than to try and cover all modes and risk generating a sample $x$ in the low-density region between them.
* **Result:** This leads to **mode collapse**. The generator $Q$ learns to produce high-quality, realistic samples, but only from one or a few modes of the true data distribution. This is a classic failure case for standard Generative Adversarial Networks (GANs), whose objective function behaves similarly to Reverse KL.



---

## Part 2: Evaluating Generative Models from Samples

For high-dimensional data like images, we almost never have access to the true data distribution $p(x)$ or the model's distribution $q(x)$. We only have a set of real samples $\{x_i\} \sim P$ and a set of generated samples $\{\hat{x}_j\} \sim Q$.

KL divergence is useless here because estimating $p(x)$ from samples in high dimensions is intractable (the "curse of dimensionality"). We need metrics that operate directly on the two sets of samples.

### Sliced Wasserstein Distance (SWD)

The **Wasserstein Distance** (or "Earth-Mover's Distance") measures the cost of "moving" the probability mass of $Q$ to match $P$. It's a powerful metric, but computationally prohibitive in high dimensions.

**The Solution:** The **Sliced Wasserstein Distance (SWD)**.
The core idea is simple:
1.  High-dimensional (e.g., 3D) distance is hard to compute.
2.  One-dimensional (1D) distance is trivial: just sort the points from both distributions and sum the distances between corresponding pairs.
3.  We can "slice" our high-D distributions into many 1D distributions by projecting them onto random lines.
4.  The SWD is the average (or max) of the 1D Wasserstein distances over many random slices.

#### Mean Sliced Wasserstein (Mean-SWD)

This is the most common variant.

**Algorithm:**
1.  Choose a large number of random 1D projections (slices), $L$. These are random unit vectors $\theta_k \in S^{d-1}$ for $k=1...L$.
2.  For each slice $\theta_k$:
    a.  Project all real samples: $x'_i = \theta_k \cdot x_i$
    b.  Project all generated samples: $\hat{x}'_j = \theta_k \cdot \hat{x}_j$
    c.  Compute the 1D Wasserstein distance $W_1(P_{\theta_k}, Q_{\theta_k})$ between the two 1D distributions of projected points (by sorting them).
3.  The **Mean-SWD** is the average of these 1D distances.

$$
SWD(P, Q) = \mathbb{E}_{\theta \sim S^{d-1}} \left[ W_1(P_\theta, Q_\theta) \right] \approx \frac{1}{L} \sum_{k=1}^L W_1(P_{\theta_k}, Q_{\theta_k})
$$

**Use:** Mean-SWD provides a robust, computationally efficient metric for comparing the overall structure of two sample sets. It's widely used for evaluating image generation quality.

#### Max Sliced Wasserstein (Max-SWD)

Instead of *averaging* over all slices, Max-SWD finds the *single slice* that maximizes the 1D Wasserstein distance.

$$
Max\text{-}SWD(P, Q) = \max_{\theta \in S^{d-1}} W_1(P_\theta, Q_\theta)
$$

**Use:** This metric finds the "worst-case scenario" projection—the direction in which the two distributions are *most* different. It is a stronger metric than Mean-SWD but requires an optimization procedure to find the maximizing $\theta$, making it more computationally expensive.

### Wasserstein using Embedding (Fréchet Distance)

A problem with SWD is that it operates in *pixel space*. A one-pixel shift in an image can result in a massive pixel-space distance, even though the image is semantically identical.

**The Solution:** Compare distributions in a more meaningful *feature space* (or "embedding space").

The most famous example is the **Fréchet Inception Distance (FID)**.

**Algorithm:**
1.  **Get Embeddings:**
    a.  Take a pre-trained deep neural network, typically **InceptionV3** (trained on ImageNet).
    b.  Pass all your *real* images $\{x_i\}$ through the network and get the feature activations from a deep layer (e.g., the final pooling layer). This gives a set of real feature vectors $\{f_i\}$.
    c.  Pass all your *generated* images $\{\hat{x}_j\}$ through the *same* network and get a set of fake feature vectors $\{\hat{f}_j\}$.

2.  **Model as Gaussians:**
    a.  Assume the real feature vectors $\{f_i\}$ are samples from a multivariate Gaussian $\mathcal{N}(\mu_r, \Sigma_r)$. Calculate the sample mean $\mu_r$ and covariance $\Sigma_r$.
    b.  Assume the fake feature vectors $\{\hat{f}_j\}$ are samples from another multivariate Gaussian $\mathcal{N}(\mu_g, \Sigma_g)$. Calculate the sample mean $\mu_g$ and covariance $\Sigma_g$.

3.  **Calculate Fréchet Distance:**
    The FID is the 2-Wasserstein distance between these two *Gaussian* distributions, which has a convenient closed-form solution:

$$
FID = ||\mu_r - \mu_g||_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

**Use:** FID is the *de facto* standard for evaluating GANs. It compares the statistics of generated images to real images in a space that understands *semantic features* (like objects, textures, and shapes) rather than just raw pixel values. A lower FID score means the distributions of real and generated features are more similar, indicating higher-quality and more diverse generated images.