---
layout: post
title: "Understanding Flow Matching for Image Generation"
---

> Nature’s voice is mathematics, its language is differential equations.

This quote, which I did once find attributed to Galileo, has bounced around scientific circles for quite some time and its sentiment has been shared by many scientists and much research in physics and chemistry has backed it up. Differential equations seem to describe much of the natural world and by using them, our power to model the world expands. This blog post aims to cover a more recent use case of this tool: image generation.

# Introduction and Intuition

To start, an analogy. Imagine you are an explorer in the 16th century getting ready to set sail across a vast ocean. You start in Europe and are setting sail to the Americas, hoping for riches. You know what Europe looks like and can start from many different points on the continent. You also have some information on some major landmarks in the Americas from the work of previous explorers. This is where you and I stand as we embark on our flow matching journey. We have some noise that is distributed according to a multivariate standard Gaussian and we can sample from this, just like how we can start our voyage from anywhere in Europe. We also have some pictures, but we don’t know the exact distribution of these images, just like how we know of some landmarks in the Americas but don’t have a great map of the continent.

Now in order to make the journey across the Atlantic, we need to follow a path. For our intrepid voyager, this means following the trade winds across the ocean. For our flow matching model, it means following a path defined by a vector field. Our model will learn this vector field, and we can use ODE solvers to trace this path and generate our output. Similarly, our captain can use a compass and map to follow the trade winds we’ve charted.

Let’s contrast this with another major architecture for training generative models: diffusion. Imagine diffusion as your rival explorer who throws caution to the wind. Instead of studying the trade winds before setting sail, they simply begin their journey and soldier through whatever bumps they encounter along the way. Our diffusion rival accepts the jaggedness of the storm, reacting to every random gust of noise, hoping to eventually stumble upon the shore.

In contrast, our flow matching explorer learns the winds themselves and charts efficient, deterministic paths from any starting point in Europe to any destination in the New World. By learning the vector field that transforms Gaussian noise into real images from our dataset, we can efficiently generate new images from new Gaussian noise.

# Plotting the Course

Now that we have the intuition down for what flow matching is aiming to do, we can get into the “how” for the process. Before we get into that, I want to clarify the terms we are using here and connect back to the analogy one last time to really hammer it home:

* **The Departure ($x_0$):** This is a point in Europe, which we define as our "source" distribution. In generative modeling, this is simply a sample of pure noise from a multivariate standard Gaussian, $p_{\text{init}} = \mathcal{N}(0, I)$.
* **The Destination ($x_1$):** This is a landmark in the New World. For us, this is a real image sampled from our dataset ($p_{\text{data}}$).
* **The Voyage Time ($t$):** A value between $0$ and $1$. $t=0$ represents the moment we leave the docks in Europe, and $t=1$ is the moment we arrive at the shore of the Americas.