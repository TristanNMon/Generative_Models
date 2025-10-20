# Langevin Sampling with Denoisers (Score-Based Generation)

This document explains the theory, motivation, and limitations of using denoisers to perform Langevin sampling, a foundational concept in modern score-based generative modeling.

---

## Part 1: The Theory

### What is Langevin Dynamics?

**Langevin Dynamics** is a Markov Chain Monte Carlo (MCMC) algorithm used to draw samples from a probability distribution $p(x)$. It's particularly useful when we only know $p(x)$ up to a constant, as it only requires the **score function**: $\nabla_x \log p(x)$.

The core idea comes from physics. Imagine $p(x)$ as a landscape where $-\log p(x)$ is the energy. We want to find samples $x$ that live in the low-energy valleys (high-probability regions).

Langevin dynamics simulates a particle moving in this landscape. It takes two actions at each step:
1.  **Gradient Step:** Move "downhill" towards higher probability, following the gradient of the log-probability (the score function).
2.  **Noise Injection:** Add a small amount of random Gaussian noise to "kick" the particle around, allowing it to explore the landscape and not get stuck in a single local minimum (mode).

The update rule for Langevin sampling is:
$$
x_{t+1} = x_t + \frac{\epsilon^2}{2} \nabla_x \log p(x_t) + \epsilon z_t
$$
where:
* $x_t$ is the sample at step $t$.
* $\epsilon$ is the step size.
* $\nabla_x \log p(x_t)$ is the **score function** of the data distribution $p$ at $x_t$.
* $z_t$ is random noise drawn from a standard normal distribution, $z_t \sim \mathcal{N}(0, I)$.

If $\epsilon \to 0$ and $t \to \infty$, the distribution of $x_t$ will converge to the true data distribution $p(x)$.

### The Problem: The Unknown Score

For complex, high-dimensional data like images, we don't have an explicit formula for $p(x)$. We only have a set of samples from it.

This means we **cannot compute the score function** $\nabla_x \log p(x)$, making standard Langevin dynamics impossible.

### The Solution: Denoising Score Matching

This is the key insight. It turns out that we can **train a neural network to estimate the score function** without ever knowing $p(x)$.

This is done through **Denoising Score Matching (DSM)**. The procedure is:
1.  **Take a data sample:** $x \sim p(x)$.
2.  **Corrupt it with noise:** $\tilde{x} = x + \sigma z$, where $z \sim \mathcal{N}(0, I)$ and $\sigma$ is a chosen noise level.
3.  **Train a denoiser:** Train a neural network, $s_\theta(\tilde{x}, \sigma)$, to predict the noise $z$ that was added. The training objective is a simple L2 loss:
    $$
    \mathcal{L} = \mathbb{E}_{x, z} \left[ ||s_\theta(x + \sigma z, \sigma) - z||^2 \right]
    $$

A key theoretical result (from Denoising Score Matching) shows that the optimal denoiser $s_\theta^*$ is directly related to the score of the *noise-perturbed* data distribution $p_\sigma(\tilde{x})$:
$$
s_\theta^*(\tilde{x}, \sigma) \approx -\sigma \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})
$$
Therefore, we have an estimate for the score:
$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) \approx -\frac{s_\theta(\tilde{x}, \sigma)}{\sigma}
$$

### The "Denoising-Langevin" Algorithm

We can now run Langevin dynamics! We replace the unknown true score $\nabla_x \log p(x)$ with our trained denoiser's estimate $\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})$.

The new update rule becomes:
$$
x_{t+1} = x_t + \frac{\epsilon^2}{2} \left( -\frac{s_\theta(x_t, \sigma)}{\sigma} \right) + \epsilon z_t
$$

This is **Langevin sampling guided by a denoiser**. We start with pure noise $x_0 \sim \mathcal{N}(0, I)$ and iteratively "denoise" it, following the estimated score, until it becomes a realistic sample.

**Refinement: Annealed Langevin Dynamics**
In practice, using a single noise level $\sigma$ doesn't work well.
* A **high $\sigma$** is good for a "blurry" score that captures the global structure and finds different modes.
* A **low $\sigma$** is good for a "sharp" score that fills in fine-grained details.

The solution is **Annealed Langevin Dynamics**. We use a *schedule* of decreasing noise levels $\sigma_1 > \sigma_2 > ... > \sigma_L$. We start sampling with the high-noise $\sigma_1$ to get the rough shape, and then gradually "anneal" to lower noise levels to refine the sample. This is the core mechanism behind **Score-Based Generative Models (SDEs)** and **Denoising Diffusion Probabilistic Models (DDPMs)**.

---

## Part 2: Motives - Why Use This Approach?

1.  **Stable Training:** The training objective (denoising) is a simple L2 regression task. This is generally much more stable and easier to optimize than the adversarial min-max game of GANs.
2.  **No Mode Collapse:** Unlike GANs, which can "forget" modes of the data, the score-matching objective forces the model to learn the score for *all* data. The noise in the sampling process also helps explore the full distribution, leading to high-diversity samples.
3.  **Explicit Likelihood (in some variants):** These models form the basis of diffusion models, which (with some extra work) can compute explicit log-likelihoods, allowing for quantitative model comparison.
4.  **Foundation for Diffusion Models:** This entire concept is the theoretical backbone of diffusion models, which are state-of-the-art in image, audio, and (increasingly) video generation.

---

## Part 3: Limitations and Challenges

1.  **Slow Sampling:** This is the most significant drawback. To get one high-quality sample, you must run the denoiser network $s_\theta$ for many steps (e.g., $T=1000$). This is 1000x slower than a single forward pass in a GAN or VAE.
2.  **Model Accuracy is Critical:** The sampler is "blind" and just follows the score function. Any errors or artifacts learned by the denoiser $s_\theta$ will be "integrated" and will appear in the final samples. If the score estimate is poor, the samples will be poor.
3.  **Complex Hyperparameters:** Annealed Langevin dynamics requires a carefully designed noise schedule ($\sigma_1, ..., \sigma_L$) and step sizes ($\epsilon_i$). A bad schedule can lead to divergent sampling or low-quality results.
4.  **Mode-Hopping Difficulty:** While better than standard GANs, the sampler can still struggle to jump between a-synchronous, well-separated modes in the data distribution, especially at low noise levels.