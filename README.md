# Generative_Models

## Theoretical Foundations

This repository contains a series of detailed guides in Markdown that build from foundational principles to state-of-the-art models:

### KL Divergence & Generative Evaluation:

- Explores the difference between Forward (mode-covering) and Reverse (mode-seeking) KL divergence when working with mixture distributions.

- Details how to evaluate generative models using sample-based metrics like Max/Mean Sliced Wasserstein Distance and Wasserstein using embeddings (Fréchet Distance).

### Langevin Sampling with Denoisers:

- Introduces the theory behind using Langevin dynamics for sampling.

- Explains the key insight of Denoising Score Matching, which allows us to learn the score function of a complex data distribution by training a simple denoiser.

### Denoising Diffusion Probabilistic Models (DDPMs):

- Provides a deep dive into the DDPM framework, explaining the forward (diffusion) and backward (denoising) processes.

- Highlights its improvements over traditional Langevin methods and breaks down the primary sources of error: initialization, training, and discretization.
