# DDPM: Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPMs) are a class of **score-based generative models** that have become state-of-the-art for high-fidelity image synthesis. They work by first "destroying" data in a fixed **forward process** (diffusion) and then "learning" a **backward process** (denoising) to reverse the destruction and generate new data from pure noise.

---

## Part 1: The Core Theory

The DDPM framework consists of two opposing processes that operate over a number of finite timesteps, $T$ (e.g., $T=1000$).

1.  **Forward Process (Diffusion):** This is a fixed, non-learned Markov process $q$. It takes a real data sample $x_0$ (e.g., a clean image) and gradually adds Gaussian noise at each step $t$, producing a sequence of progressively noisier samples $x_1, x_2, ..., x_T$. The noise schedule $\beta_t$ is fixed, so $x_T$ is indistinguishable from pure Gaussian noise.

2.  **Backward Process (Generation):** This is a learned Markov process $p_\theta$. It starts with pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$ and learns to "denoise" it step-by-step: $x_{T-1}, x_{T-2}, ...$ until it produces a clean data sample $x_0$. This reverse process is guided by a neural network that is trained to undo one step of the diffusion.



---

## Part 2: Improvement on Langevin Denoising

DDPMs are a specific refinement of the broader idea of **annealed Langevin dynamics** (which uses a denoiser to estimate the data's score $\nabla_x \log p_t(x_t)$ and samples via an MCMC process).

* **Annealed Langevin Dynamics (ALD):** This is an MCMC sampling procedure. The update rule involves both a "drift" term (from the score network) and an "injected noise" term at *every* step to ensure the sampler explores the distribution.
    $$
    x_{t-1} = x_t + \frac{\epsilon}{2} \nabla_x \log p_t(x_t) + \sqrt{\epsilon} z_t
    $$
    This can be sensitive to the step size $\epsilon$ and the noise schedule.

* **DDPM Improvement:** DDPMs re-parameterize the entire reverse process. Instead of just learning the score, the neural network learns to predict the parameters of the conditional distribution $p_\theta(x_{t-1} | x_t)$.
    * It's proven that if the forward step noise $\beta_t$ is small, this reverse conditional $q(x_{t-1} | x_t, x_0)$ is also a Gaussian.
    * The DDPM network $D_\theta(x_t, t)$ is trained to predict the **noise** ($\epsilon$) that was added to get $x_t$.
    * From this predicted noise $\hat{\epsilon}$, we can analytically compute the *mean* of the reverse Gaussian $p_\theta(x_{t-1} | x_t)$. The variance is typically fixed.
    * The sampling step becomes:
        1.  Predict $\hat{\epsilon} = D_\theta(x_t, t)$.
        2.  Calculate the mean of $x_{t-1}$: $\tilde{\mu}_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right)$.
        3.  Sample $x_{t-1} \sim \mathcal{N}(\tilde{\mu}_\theta, \tilde{\beta}_t I)$.

This formulation as a generative *chain* (not an MCMC sampler) is often more stable, easier to train (with a simple L2 noise-prediction loss), and leads to the exceptionally high sample quality that made DDPMs famous.

---

## Part 3: The Forward Process (Diffusion)

The forward process $q$ is defined as a fixed Markov chain that adds small amounts of Gaussian noise at each step $t$ according to a variance schedule $\beta_1, ..., \beta_T$.

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

This says $x_t$ is a scaled version of $x_{t-1}$ plus some noise.

A key property of this process is that we can sample $x_t$ at any arbitrary timestep $t$ *directly* from the original $x_0$, without iterating through all the steps. This is crucial for efficient training.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. The "one-shot" forward process formula is:
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$
Or, more simply:
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{where} \; \epsilon \sim \mathcal{N}(0, I)
$$

As $t \to T$, $\bar{\alpha}_T \to 0$, which means $x_T \approx \mathcal{N}(0, I)$. The data is completely destroyed into pure noise.

---

## Part 4: The Backward Process (Generation)

The backward process $p_\theta$ is our generative model. It learns to reverse the diffusion, starting from $x_T \sim \mathcal{N}(0, I)$ and sampling $x_{t-1} \sim p_\theta(x_{t-1} | x_t)$ until it reaches $x_0$.

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)
$$

The core task is to model $p_\theta(x_{t-1} | x_t)$. We model this as a Gaussian whose variance is fixed and whose mean $\tilde{\mu}_\theta(x_t, t)$ is predicted by a neural network.

The **training objective** is derived from the Variational Lower Bound (ELBO) on the data likelihood. It simplifies to a very simple L2 loss:
$$
\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - D_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)||^2 \right]
$$
This just means:
1.  Pick a real image $x_0$.
2.  Pick a random time $t$.
3.  Generate a random noise $\epsilon$.
4.  Create the noisy image $x_t$ using the one-shot formula.
5.  Feed $x_t$ and $t$ to the network $D_\theta$.
6.  Train the network to **predict the original noise $\epsilon$** that was added.

---

## Part 5: Behavior on a Mixture of Gaussians

DDPMs are excellent at handling multi-modal distributions, like a Gaussian Mixture Model (GMM), and do not suffer from the "mode collapse" seen in many GANs.

* **Forward:** In the forward process, samples from all modes (e.g., two distinct clusters) are gradually noised. As $t$ increases, the modes "smear" and "merge," eventually collapsing into a single, unimodal $\mathcal{N}(0, I)$ distribution.
* **Backward:** The generative process starts from this single $\mathcal{N}(0, I)$ prior. As the denoiser network $D_\theta$ reverses the time $t$, it implicitly learns the score function $\nabla_x \log p_t(x)$. For a GMM, the score function at intermediate $t$ values will "point away" from the low-probability region *between* the modes and "point towards" the centers of the modes.
* **Result:** A sampling trajectory $x_t$ that starts in the middle will be pushed by the score field into one of the modes. Running the sampler many times will produce samples that cover all modes in their correct proportions.



---

## Part 6: How DDPMs Generate Images

The theory applies directly to images, where $x_0$ is just a high-dimensional vector (or tensor) of pixels.

1.  **The Network ($D_\theta$):** The denoiser network is almost always a **U-Net**. A U-Net architecture is ideal because:
    * It takes an image (the noisy $x_t$) as input and outputs an image (the predicted noise $\hat{\epsilon}$) of the *exact same dimensions*.
    * Its encoder-decoder structure with skip-connections allows it to capture features at multiple scales, which is perfect for separating high-frequency noise from low-frequency image content.

2.  **Time Conditioning:** The network must know *how much* noise to remove. The timestep $t$ is fed into the network (usually encoded via sinusoidal "positional embeddings") so that it can adapt its denoising strategy.

3.  **The Generation Process:**
    * **Step 1:** Sample a tensor $x_T$ of pure Gaussian noise from $\mathcal{N}(0, I)$.
    * **Step 2:** For $t = T, T-1, ..., 1$:
        * Feed $x_t$ and the time $t$ into the U-Net $D_\theta$.
        * The network outputs its prediction of the noise, $\hat{\epsilon}$.
        * Use this $\hat{\epsilon}$ to calculate the mean $\tilde{\mu}_\theta(x_t, t)$ of the $x_{t-1}$ distribution.
        * Sample $x_{t-1} \sim \mathcal{N}(\tilde{\mu}_\theta, \tilde{\beta}_t I)$.
    * **Step 3:** The final $x_0$ is the generated image.

---

## Part 7: Sources of Error

There are three primary sources of error that prevent the generative model $p_\theta(x_0)$ from
perfectly matching the true data distribution $q(x_0)$.

### 1. Initialization Error
This is the error from using a finite number of steps $T$. It's the mismatch between the distribution of our fully-noised data $q(X_T)$ and the prior $\mathcal{N}(0, I)$ we actually sample from at the start of generation. (Note: the formula below uses a general prior for continuous-time models).

$$
\propto W_2(\operatorname{Law}(X_T), \mathcal{N}(0, (\sigma_T^2 + 1)Idm))
$$

### 2. Training Error
This is the error from our neural network $D_\theta$ not being a perfect denoiser. It's the gap between our model's prediction and the "true" denoised $X_0$. (Note: this formula assumes the network $D$ is trained to predict $X_0$, an alternative but equivalent parameterization to predicting noise $\epsilon$).

$$
\propto \sum_{t=0}^{T}\mathbb{E}[\|D(X_t, \sigma_t) - X_0\|^2]
$$

### 3. Discretization Error
This error arises because we are approximating a continuous-time diffusion process with a finite number of discrete steps ($T$). The larger the "jump" between noise levels $\sigma_t$ and $\sigma_{t+1}$, the larger the error.

$$
\propto (1 - \frac{\sigma_t^2}{\sigma_{t+1}^2})
$$