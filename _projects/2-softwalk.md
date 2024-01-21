---
layout: post
title: Fairness-Enhanced Representational Learning 
description: Introducing SoftWalk. Significantly improving fairness and node representational quality.
image:  /assets/images/thumbnail-0.png
---


> The work was accepted for the 2022 Machine Learning Reproducibility Challenge and published in ReScience C journal and was awarded a $10,000 grant from Kaggle.
> **Paper links**: [OpenReview](https://openreview.net/forum?id=tpk45Zll8eh) | [Kaggle](https://www.kaggle.com/code/lucapantea1/reproducibility-study-of-crosswalk) | [GitHub](https://github.com/Dawlau/FACT-AI)

![Header Image](/assets/images/FACT.png)


Recently, alongside [Andrei](https://github.com/Dawlau), I ventured into the realm of fairness in machine learning by getting our hands dirty with reproducing "CrossWalk: Fairness-enhanced Node Representation Learning." ([paper link](https://arxiv.org/abs/2105.02725)). The potential for machine learning systems to amplify social inequities and unfairness is receiving increasing popular and academic attention. However, there is little work enhancing fairness in graph algorithms. Even less in reproducing these algorithms, and verifying their claims...


Our starting point was the CrossWalk repository that the original authors were kind enough to share. Step $0$ to $1$ was done. However, $1$ to $n$ was a whole different story. The lack of documentation and some missing logic meant we had to roll up our sleeves and re-code from scratch. We packaged all our hard work into a neat Python package, so anyone could easily replicate the results and join in on the fun (here's the [repo link](https://github.com/Dawlau/FACT-AI)).

This is what we managed to do in a month's time:

* <span style="color: #79ba8d;font-weight:900">[Reproducibility Study]</span> Reproducing the results from the original paper: We were
able to partially reproduce the claim that Crosswalk enhances fairness, namely for
the tasks of node classification and influence maximization. The second claim,
which is that the proposed method preserves the higher‐order proximity of the
graph was successfully reproduced.
* <span style="color: #79ba8d;font-weight:900">[Extended Work]</span> Improvement of the original code: The original code is not easily runnable, so we had to refactor it and implement our own Bash and Python
scripts such that the experiments can be more easily reproduced. Further, we provide a master Python script that allows for the reproducibility of the experiments
presented in this report by running only a single terminal command.
* <span style="color: #79ba8d;font-weight:900">[Extended Work]</span> Ablation study: CrossWalk does not usually yield the expected
results out of the box, but by carefully picking the right hyperparameters we managed to make CrossWalk behave as presented in the original paper.
* <span style="color: #79ba8d;font-weight:900">[Extended Work]</span> CrossWalk visualization: We perform additional visualizations
of the edge reweighting procedure and random walk trajectories to investigate the
claims proposed by the original authors.
* <span style="color: #79ba8d;font-weight:900">[Proposed Enhancement]</span> Soft Self‐Avoiding CrossWalk: We propose an extension
to the algorithm proposed by Khajehnejad et al. which yields better node representations and higher fairness.

Quite a bit :) let's dive into it.


### The Original Model's Inner workings
<span style="color: #79ba8d;font-weight:900">CrossWalk</span> is a re-weighting method that is used to bias random walk-based algorithms towards visiting multiple groups, which in turn enhances fairness. This is mainly done by re-weighting the probabilities associated with the graph edges as follows:

* CrossWalk increases the weights of edges <span style="color: #79ba8d;font-weight:900">near the group boundaries</span>.
* CrossWalk increases the weights of edges <span style="color: #79ba8d;font-weight:900">that connect nodes from different groups</span>. 

The mathematical formulas that describe the re-weightings from points 1 and 2 are the following: 


$$
\begin{equation}
w_{vu}^{'} = \begin{cases} w_{vu}(1-\alpha) \times \frac{m(u)^p}{\sum\limits_{z \in N_v} w_{vz}m(z)^p} & \text{if } u \in N(v), l_v = l_u \\
w_{vu}\alpha \times \frac{m(u)^p}{|R_{v}| \times \sum\limits_{z \in N_{v}^c} w_{vz}m(z)^p} & \text{if } u \in N(v), l_v \neq l_u = c.\end{cases}
\end{equation}
$$

where $w_{vu}$ is the initial weight of the edge $(u,v)$, $\alpha$ and $p$ are hyperparameters, $R_{v}$ the set of groups in the neighbourhood of $v$, $l_{x}$ the group that node $x$ is part of, and $m(z)$ is the proximity of a node as described in the original paper.

Cool 🎉. However, upon further in-depth study, we realised that this re-weighting caused some subtle, yet impactful issues. 

In a nutshell, this re-weighting mechanism caused the random walks to <span style="color: #79ba8d;font-weight:900">travel back and forth between edges localted at group peripherals, esentially getting stuck in a loop</span>. This is because the re-weighting mechanism is not self-avoiding, meaning that the random walks can visit the same edge multiple times. This is a problem because the random walks are supposed to be unbiased, and the re-weighting mechanism biases these trajectories in an undesirable way.


Through sampling and visually analyzing trajectories, we explored CrossWalk's mechanics. The image at the beginning the page compares edge weights and a trajectory between a random walk algorithm without and with CrossWalk’s re-weighting. Lighter edges indicate higher edge weights, different node colors represent distinct groups, and a bolded blue line traces a sampled random walk trajectory. CrossWalk on the right visits fewer new nodes compared to the un-reweighted algorithm on the left, as the re-weighting mechanism leads the random walk into a repetitive loop, revisiting the same edges.

As a result, <u>the structural properties of the graph are not fully utilized</u>, leading to decreased representation quality.

### Our Enhancement!

We propose a solution based on Self‐Avoiding Walks  which steers the random walks towards avoiding previously visited edges by using a discounting function, parameterized by $\gamma$:

$$
\begin{equation}
P(u_i = u \lvert u_{i-1} = v) = \left\{
        \begin{array}{ll}
            \pi_{uv} \cdot \gamma^{c(u,v)}  & \quad \text{if} \ (u, v) \in E \\
            0 & \quad \text{otherwise}
        \end{array}
    \right.
\end{equation}
$$

Where $c(u, \ v)$ stores the number of times the walk has traversed the edge between node $u$ and $v$. As $\gamma \in [0, \ 1]$, if $\gamma \rightarrow 1$, the behaviour is similar to CrossWalk, while $\gamma \rightarrow 0$ stimulates the random walk to avoid already visited edges. $\pi_{uv}$ represents probability of selecting the edge $(u, \ v)$, weighted by the modifed edge weights of CrossWalk, as expressed above. We call this method <span style="color: #79ba8d;font-weight:900">SoftWalk</span>.

Our implementation uses a HashMap initialized per random walk for storing the counts of the previously visited edges, which has a worst-case time and space complexity of $\mathcal{O}(K)$, where $K$ represents the preset random walk length. Thus, our proposed method scales with $K$, which <span style="color: #79ba8d;font-weight:900">supports scalability</span>, as $K$ $<<$ $\lvert E\lvert$, where $E$ is the set of graph edges for a given graph.


### Our Results and a Bit of Discussion

The results indicate that the proposed extension <span style="color: #79ba8d;font-weight:900">consistently outperforms</span> the original implementation of CrossWalk on the two reproducible tasks - Link Prediction and Node Classification. <span style="color: #79ba8d;font-weight:900">Softwalk not only achieves better accuracy and influence percentage</span> but <span style="color: #79ba8d;font-weight:900">additionally minimizes the disparity</span> (thus maximizing fairness) for each task. Check [our paper](https://openreview.net/forum?id=tpk45Zll8eh) to see the numbers and plots!



### Limitations

Our contributions suffer from three main drawbacks. First, due to time constraints, we were unable to resolve the outstanding issues regarding link predictions, yet we were able to obtain results that follow the same trends for the other tasks. Second, the re-implementation of the original base significantly decelerated the reproduction process. And finally, the proposed approach is independent of the underlying graph characteristics, which encourages future work in the direction of estimating $\gamma$ via a non-linear function approximation (i.e. neural network-based approaches). 

<br>

___
If you made it this far, I hope you enjoyed the read and learned something new! If you have any questions, feel free to reach out to me on [Twitter](https://twitter.com/luca_pantea) or [LinkedIn](https://www.linkedin.com/in/luca-pantea/). I'd love to hear your thoughts! 😊
