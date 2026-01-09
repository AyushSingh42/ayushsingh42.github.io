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
 
 ## Plotting the Probability Path

 The probability path is another important concept to understand. If the flow is the solution for one set of initial conditions, think of the probability path $p_t$ as the satellite view of all the flows. A map that shows the probability path would show us all of the ships that started in Europe and would show the path that they all take across the ocean to each of their respective final destinations.

 In more technical terms, while the flow $\psi_t$ tracks the journey of an individual point, the probability path $p_t$ describes the global evolution of the data distribution over time. At $t=0$, our satellite view shows a cloud of ships distributed as Gaussian noise; by $t=1$, that cloud has shifted and morphed to match the complex structure of our real image dataset. We can write this out as a training objective for our neural network:

 $$
 \mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)} \| v^{\theta}_t(x) - u^{target}_t(x) \|^2
 $$

 Here, $v^{\theta}_t(x)$ is the neural network that we train that learns the global/marginal probability path. $u^{target}_t(x)$ is the real marginal probability path. Also note that we use the shorthand $\mathbb{E}_{t,z,x}[\cdot] = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}[\cdot]$ for brevity but that proofs will not be using shorthand.

 The challenge is that the real global probability path is incredibly complex and mapping it is an intractable problem. In order to circumvent this, we use conditional flow matching (CFM).

Instead of trying to map the entire ocean at once, we "cheat" by looking at individual voyages between a specific noise sample $x_0$ and a specific target image $x_1$. This allows us to define a conditional probability path, which is simply a straight line in the space $\mathbb{R}^d$. This is known as linear interpolation:

$$
x_t = (1 - t)x_0 + t x_1
$$

By choosing this linear path, we ensure that the ships are moving using the shortest possible distance. This results in a constant velocity:

$$\frac{d}{dt}x_t = x_1 - x_0$$

This constant value $(x_1 - x_0)$ becomes the "target wind" for our neural network. Rather than reacting to the jagged, random gusts of a diffusion model, our navigator learns to follow these steady, straight-line trade winds. As it learns thousands of these individual straight-line voyages, the model naturally reconstructs the global map of the entire ocean.

This last part is game changing. Just by learning a bunch of conditional probability paths, we are able to fully circumvent the intractable problem of learning the marginal probability path. We can define our conditional loss as follows:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_0(x_0)} \| v^{\theta}_t(x_t) - (x_1 - x_0) \|^2$$

The proof for why $L_{FM} = L_{CFM}$ is quite simple and I highly recommend you go through it. If you don't care for the math and just want to get to the code, then that's also an option.

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

# The Navigator's Code

While the math of vector fields and intractable marginals can feel daunting, the beauty of Flow Matching is how it simplifies the implementation. In a typical Diffusion model, you have to manage complex noise schedules and stochastic differential equations. In Flow Matching, the training loop is a simple, deterministic recipe.

Below is a minimal PyTorch implementation of the training objective we derived:

```python
def flow_matching_loss(model, x1):
    """
    x1: Real data sample (The Americas)
    model: Neural network v_theta(x, t)
    """
    # 1. Sample pure noise (Departure from Europe)
    x0 = torch.randn_like(x1)
    
    # 2. Pick a random moment in the voyage
    t = torch.rand(x1.shape[0], 1, 1, 1).to(x1.device)
    
    # 3. Compute our position on the straight-line path (Linear Interpolation)
    xt = (1 - t) * x0 + t * x1
    
    # 4. Our target is the constant velocity (The Wind)
    ut = x1 - x0
    
    # 5. Predict the velocity and compute MSE
    vt = model(xt, t)
    return torch.mean((vt - ut) ** 2)
```

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