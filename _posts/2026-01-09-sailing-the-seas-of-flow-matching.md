---
layout: post
title: "Sailing the Seas of Flow Matching"
---

> Nature’s voice is mathematics, its language is differential equations.

This quote, which I did once find attributed to Galileo, has bounced around scientific circles for quite some time and its sentiment has been shared by many scientists and much research in physics and chemistry has backed it up. Differential equations seem to describe much of the natural world and by using them, our power to model the world expands. This blog post aims to cover a more recent use case of this tool: image generation.

# Introduction and Intuition

To start, an analogy. Imagine you are an explorer in the 16th century getting ready to set sail across a vast ocean. You start in Europe and are setting sail to the Americas, hoping for riches. You know what Europe looks like and can start from many different points on the continent. You also have some information on some major landmarks in the Americas from the work of previous explorers. This is where you and I stand as we embark on our flow matching journey. We have some noise that is distributed according to a multivariate standard Gaussian and we can sample from this, just like how we can start our voyage from anywhere in Europe. We also have some pictures, but we don’t know the exact distribution of these images, just like how we know of some landmarks in the Americas but don’t have a great map of the continent.

Now in order to make the journey across the Atlantic, we need to follow a path. For our intrepid voyager, this means following the trade winds across the ocean. For our flow matching model, it means following a path defined by a vector field. Our model will learn this vector field, and we can use ODE solvers to trace this path and generate our output. Similarly, our captain can use a compass and map to follow the trade winds we’ve charted.

Let’s contrast this with another major architecture for training generative models: diffusion. Imagine diffusion as your rival explorer who throws caution to the wind. Instead of studying the trade winds before setting sail, they simply begin their journey and soldier through whatever bumps they encounter along the way. Our diffusion rival accepts the jaggedness of the storm, reacting to every random gust of noise, hoping to eventually stumble upon the shore.

In contrast, our flow matching explorer learns the winds themselves and charts efficient, deterministic paths from any starting point in Europe to any destination in the New World. By learning the vector field that transforms Gaussian noise into real images from our dataset, we can efficiently generate new images from new Gaussian noise.

# Preparing for the Journey
A quick note, this section will assume some level of familiarity with differential equations.

Now that we have the intuition down for what flow matching is aiming to do, we can get into the “how” for the process. Before we get into that, I want to clarify the terms we are using here and connect back to the analogy one last time to really hammer it home:

* **$x_0$:** This is simply a sample of pure noise from a multivariate standard Gaussian, $p_{\text{init}} = \mathcal{N}(0, I)$. In our analogy, this is the point in Europe that we set sail from.
* **$x_1$:** This is a real image sampled from our dataset ($p_{\text{data}}$). This was the landmark that we knew of in the Americas in our analogy.
* **$t$:** A value between $0$ and $1$. $t=0$ represents the moment we leave the docks in Europe, and $t=1$ is the moment we arrive at the shore of the Americas. In our generative process, $t=0$ is the the time at which our image is pure Gaussian noise, $t=1$ is when our image is a real image from $p_{data}$, and every in between step is some version of the image with less noise.

Now our goal is to model the winds that will blow any ship from that point $x_0$ to the real image $x_1$. This model of the wind is a vector field, $u_t$, and the particular path that we want to follow is called a flow, $\psi_t$.

Before we set sail, let's stock the ship with the necessary import statements:

```python
# pip install torch torchvision

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
```

Let's also plan out some of our voyage. It's always good to be prepared. Here we define some key hyperparameters:

```python
@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    lr: float = 2e-4
    epochs: int = 5
    num_workers: int = 2

    # Flow matching params
    sigma_min: float = 1e-2  # small endpoint noise
    t_eps: float = 1e-5      # avoid exact endpoints

    # Sampling params
    sample_steps: int = 50   # RK4 steps from t=0->1
    n_samples: int = 64
    out_dir: str = "fm_out"

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)
print("Device:", cfg.device)
```

This configuration encodes the practical constraints of our voyage. The learning rate controls how quickly the navigator updates its internal map of the winds, while the batch size determines how many ships we observe at once. The parameter $\sigma_{\min}$ enforces a small amount of uncertainty at landfall, preventing the winds from becoming singular near the destination. Finally, the sampling parameters determine how finely we trace the learned route during inference.

## A Quick Differential Equations Aside

The solution to an ODE is defined by a trajectory that maps some time $t$ to some location in the space $\mathbb{R}^d$.
$$
X: [0, 1] \rightarrow \mathbb{R}^d, t \mapsto X_t
$$
Every ODE is defined by some vector field. Going back to the analogy, the vector field is all of the winds that blow across the Atlantic. The solution mentioned above is one such path along one current that can take one ship across the ocean. We write out an ODE defined by a vector field as follows:
$$
\frac{\text{d}}{\text{d}t}X_t = u_t(X_t)
\\
X_0 = x_0
$$
The top line says that our ODE is defined according to some vector field $u_t$ with some initial conditions $x_0$. The derivative of $X_t$ is given by the direction of the vector field. In order to find the flow $\psi_t$, we need some initial conditions $X_0 = x_0$ at some $t = 0$. The flow will tell us the current state when we plug in some time $t$. In our analogy, it would allow us to know exactly where the ship is at some time $t$ just by knowing the initial conditions of the ship. This is written out as follows:

$$\psi : \mathbb{R}^d \times [0, 1] \mapsto \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0)$$
$$\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0) = u_t(\psi_t(x_0))$$
$$\psi_0(x_0) = x_0$$

> **A note on $\sigma_{\min}$.**  
> In practice, flow matching models do not force trajectories to land exactly on a data point at $t=1$. Instead, the endpoint distribution is taken to be a narrow Gaussian centered at $x_1$, with variance $\sigma_{\min}^2 I$. This small but nonzero noise level prevents singular behavior in the vector field, stabilizes training, and improves numerical behavior during ODE sampling. In the idealized limit $\sigma_{\min} \to 0$, the target velocity reduces to the intuitive expression $x_1 - x_0$, which we use throughout for intuition.

 
## Plotting the Probability Path

So far, we have focused on the journey of a single ship. One starting point $x_0$, one destination $x_1$, and one path through the ocean defined by a vector field. But to train a generative model, we must zoom out.

The *flow* $\psi_t$ describes how one individual point moves over time. The *probability path* $p_t$, on the other hand, is the satellite view. It captures how an entire distribution of ships evolves as time progresses. At $t=0$, this distribution is a diffuse cloud of ships spread across Europe, sampled from a standard Gaussian. By $t=1$, that cloud has condensed and reshaped itself into the complex structure of real images from our dataset.

Formally, $p_t$ is the marginal distribution induced by transporting noise samples forward in time under the learned vector field. If we could directly compute the true vector field that governs this global evolution, we could train a neural network by minimizing

$$
\mathcal{L}_{\text{FM}}(\theta)
= \mathbb{E}_{t,\,x \sim p_t}
\big\| v_t^\theta(x) - u_t^{\text{target}}(x) \big\|^2.
$$

Here, $v_t^\theta(x)$ is our neural network and $u_t^{\text{target}}(x)$ is the true velocity field that moves the entire distribution $p_t$ through time.

Unfortunately, this global wind map is extraordinarily complex. Modeling it directly would require knowing the full intermediate distributions between Gaussian noise and real images, which is computationally intractable.

### Conditional Flow Matching

Instead of mapping the entire ocean at once, flow matching takes a more clever approach. We condition on a specific destination $x_1$ and only consider voyages that end at that point. In other words, rather than learning the marginal probability path $p_t$, we define a *conditional probability path* $p_t(\cdot \mid x_1)$.

In practice, this conditional path is chosen to be simple. A common and effective choice is a Gaussian whose mean moves toward $x_1$ while its variance shrinks over time. Early in the journey, ships are widely dispersed. As $t$ increases, they become more concentrated around their destination.

For a particular and important case, known as the optimal transport (OT) path, the corresponding flow map takes a particularly elegant form:

$$
\psi_t(x_0)
= \big(1 - (1 - \sigma_{\min}) t\big) x_0 + t x_1.
$$

This path is linear in time and induces a *constant velocity field*

$$
\frac{d}{dt}\psi_t(x_0)
= x_1 - (1 - \sigma_{\min}) x_0.
$$

When $\sigma_{\min}$ is small, this velocity closely resembles a straight push from noise toward data. In the idealized limit where $\sigma_{\min} \to 0$, it reduces to the intuitive expression $x_1 - x_0$.

This constant velocity becomes the regression target for our neural network. Rather than learning a complex, stochastic process, the model simply learns the steady winds that carry ships from their noisy origins to their destinations.

As it learns thousands of these conditional voyages, each corresponding to a different destination image, the model implicitly reconstructs the global probability path. This is the key insight of conditional flow matching: by learning many simple, local maps, we recover the full global transport without ever modeling it directly.

The resulting training objective is

$$
\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}_{t,\,x_1 \sim p_{\text{data}},\,x_0 \sim p_0}
\big\| v_t^\theta(x_t) - \big(x_1 - (1 - \sigma_{\min}) x_0\big) \big\|^2,
$$

and remarkably, this loss is equivalent to the original flow matching objective up to an additive constant.


<details>
<summary>Proof for why Flow Matching equals Conditional Flow Matching</summary>

$$\begin{aligned}
\mathbb{E}_{t\sim\text{Unif}, x\sim p_t}[u_t^{\theta}(x)^T u_t^{\text{target}}(x)] 
&\stackrel{(i)}{=} \int_0^1 \int p_t(x) u_t^{\theta}(x)^T u_t^{\text{target}}(x) \, dx \, dt \\
&\stackrel{(ii)}{=} \int_0^1 \int p_t(x) u_t^{\theta}(x)^T \left[ \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \, dz \right] dx \, dt \\
&\stackrel{(iii)}{=} \int_0^1 \int \int u_t^{\theta}(x)^T u_t^{\text{target}}(x|z) p_t(x|z) p_{\text{data}}(z) \, dz \, dx \, dt \\
&\stackrel{(iv)}{=} \mathbb{E}_{t\sim\text{Unif}, z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}[u_t^{\theta}(x)^T u_t^{\text{target}}(x|z)]
\end{aligned}$$

$$\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) 
&\stackrel{(i)}{=} \mathbb{E}_{t, z, x}[\|u_t^{\theta}(x)\|^2] - 2\mathbb{E}_{t, z, x}[u_t^{\theta}(x)^T u_t^{\text{target}}(x|z)] + C_1 \\
&\stackrel{(ii)}{=} \mathbb{E}_{t, z, x}[\|u_t^{\theta}(x)\|^2 - 2u_t^{\theta}(x)^T u_t^{\text{target}}(x|z) + \|u_t^{\text{target}}(x|z)\|^2 - \|u_t^{\text{target}}(x|z)\|^2] + C_1 \\
&\stackrel{(iii)}{=} \mathbb{E}_{t, z, x}[\|u_t^{\theta}(x) - u_t^{\text{target}}(x|z)\|^2] + \underbrace{\mathbb{E}_{t, z, x}[-\|u_t^{\text{target}}(x|z)\|^2]}_{C_2} + C_1 \\
&\stackrel{(iv)}{=} \mathcal{L}_{\text{CFM}}(\theta) + C_2 + C_1 \\
&:= \mathcal{L}_{\text{CFM}}(\theta) + C
\end{aligned}$$

</details>

<br>

# The Navigator’s Code

The beauty of Flow Matching is that the training loop is simple and deterministic.

The first question the navigator must answer is what time it is. The optimal direction to steer depends not only on where the ship is, but also on how far along the voyage it has progressed. Near Europe, the winds behave differently than they do near the Americas.

To make time usable by a neural network, we embed the scalar $t \in [0,1]$ into a higher-dimensional representation using sinusoidal features.


```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, 6, half, device=t.device))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

Think of this as giving the ship a clock. The same location in the Atlantic can demand different steering depending on whether we just left Europe or we are nearing the Americas. The time embedding turns the scalar $t$ into a rich signal the network can use to adjust its winds.

Time enters the model as a continuous signal rather than a discrete step count. The sinusoidal embedding allows the navigator to smoothly interpolate its behavior across the voyage, much like how seasonal winds change gradually rather than abruptly. This choice ensures that nearby times correspond to nearby representations, which is essential for stable ODE integration.

Next, we construct the navigator itself. This neural network represents the learned vector field $v_\theta(x,t)$. Given the ship’s current position $x$ and the current time $t$, it outputs a velocity vector indicating which direction to move next.

In physical terms, this network predicts the local trade winds at every point in space and time.

Next, we construct the navigator itself. This neural network represents the learned vector field $v_\theta(x,t)$. Given the ship’s current position $x$ and the current time $t$, it outputs a velocity vector indicating which direction to move next.

In physical terms, this network predicts the local trade winds at every point in space and time.

```python
class VelocityNet(nn.Module):
    def __init__(self, ch=64, tdim=128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.mlp = nn.Linear(tdim, ch)

        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.mlp(self.time(t))[:, :, None, None]
        return self.net(x) + t_emb
```

This network is our navigator. At each point in space and time, it predicts a velocity vector indicating how the ship should move next. Conditioning on both position and time allows the same navigator to guide ships at every stage of the journey, from open ocean to final approach. Architecturally, we keep the model simple to emphasize that Flow Matching’s power comes from the training objective, not architectural complexity.

During training, we do not simulate entire voyages from start to finish. Instead, we randomly stop ships at intermediate times and ask: *what winds should be acting here?* This is the essence of Conditional Flow Matching.

The function below samples a departure point $x_0$, a destination $x_1$, and a random time $t$, then computes the corresponding point $x_t$ along the optimal transport path and its constant target velocity.


```python
def sample_ot_path(x1):
    B = x1.size(0)
    t = torch.rand(B, device=x1.device)
    x0 = torch.randn_like(x1)

    a = 1 - (1 - cfg.sigma_min) * t
    xt = a[:, None, None, None] * x0 + t[:, None, None, None] * x1
    u = x1 - (1 - cfg.sigma_min) * x0
    return t, xt, u
```

Rather than simulating entire voyages during training, we sample random checkpoints along many possible routes. This function constructs those checkpoints by pairing a departure point, a destination, and a random time, then computing the corresponding position and constant OT velocity. In nautical terms, we pause ships mid-voyage and ask: “what wind should be blowing here if the ship is to reach land efficiently?”


With the target wind in hand, training reduces to a simple regression problem. The navigator predicts a velocity at $(x_t, t)$, and we penalize deviations from the true OT velocity. Over many such partial voyages, the model learns a globally consistent wind map.


```python
def flow_matching_loss(model, x1):
    t, xt, u = sample_ot_path(x1)
    v = model(xt, t)
    return ((v - u) ** 2).mean()
```

Training reduces to a simple regression problem. The navigator proposes a wind at the current position and time, and we penalize deviations from the optimal transport wind. By repeating this process across countless partial voyages, the model gradually reconstructs a globally consistent wind map without ever observing the full journey at once.


The training loop repeatedly sends fleets of ships across random segments of the ocean. Each batch corresponds to many independent voyages, all sharing the same navigator. Over time, the navigator becomes increasingly accurate at predicting which winds will carry ships toward land.


```python
def train():
    data = datasets.MNIST(
        ".", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
    )
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)

    model = VelocityNet().to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for _ in range(cfg.epochs):
        for x1, _ in loader:
            x1 = x1.to(cfg.device)
            loss = flow_matching_loss(model, x1)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
```

This loop represents many seasons of exploration. Each batch corresponds to a fleet of ships, each heading toward different destinations and observed at different times. Over repeated epochs, the navigator refines its understanding of the winds until it can reliably guide ships from noise to data across the entire ocean.


Once training is complete, generation is simply navigation. We release new ships from Europe (fresh Gaussian noise) and numerically integrate the learned vector field forward in time. This deterministic integration replaces the many stochastic correction steps used in diffusion models.

```python
@torch.no_grad()
def sample(model):
    x = torch.randn(cfg.n_samples, 1, 28, 28, device=cfg.device)
    dt = 1.0 / cfg.sample_steps

    for i in range(cfg.sample_steps):
        t = torch.full((x.size(0),), i * dt, device=x.device)
        x = x + dt * model(x, t)

    return (x.clamp(-1, 1) + 1) / 2
```

Viewed end to end, Flow Matching transforms generative modeling into a problem of learning and following winds. By replacing stochastic correction with deterministic navigation, we gain smoother trajectories, fewer inference steps, and a clearer conceptual link between learning and generation.


# Why Flow Matching over Diffusion?

You might wonder why we went through the trouble of charting these trade winds when diffusion is already so popular. The benefits come when it is time to actually set sail (inference).

Because we trained our model on straight-line paths, the resulting trajectories are incredibly smooth. In practical terms, this means you can generate high-quality images in far fewer steps. While Diffusion often requires dozens or hundreds of "corrections" to reach the data distribution, Flow Matching can often reach the shore in 10–20 steps using a simple Euler ODE solver.

Next, diffusion is inherently noisy; every time you generate an image, you are at the mercy of the storm. Flow Matching is a Continuous Normalizing Flow. Once you pick your starting point $x_0$, the path to the final image is deterministic. This makes the model easier to debug and allows for smooth interpolation between different images.

Finally, we have successfully traded complex variance-preserving schedules for a simple line: $x_t = (1-t)x_0 + tx_1$. This simplicity makes Flow Matching much easier to scale to massive datasets and complex architectures.

## References
For those who wish to dive deeper into the technical proofs and the broader implications of Flow Matching, I highly recommend the following papers:

Lipman, Y., Chen, R. T., Locatello, F., Esser, P., & Le, M. (2022). Flow Matching for Scalable Generative Models. arXiv preprint arXiv:2210.02747. (The foundational paper introducing the framework).

Albergo, M. S., & Vanden-Eijnden, E. (2022). Building Normalizing Flows with Stochastic Interpolants. arXiv preprint arXiv:2209.15571. (Introduces the similar concept of "Stochastic Interpolants").

Liu, X., Gong, Z., & Liu, Q. (2022). Flow Straight and Fast: Learning to Generate Images with Diffusion Exponential Integrator Sampler. arXiv preprint arXiv:2209.03003. (Often referred to as "Rectified Flow").

Cai, S., Ma, Y. A., & Liu, Q. (2024). Flow Matching: A Tutorial. arXiv preprint arXiv:2412.06264. (A more recent, pedagogical breakdown of the topic).