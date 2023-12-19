This repo centralizes different data science projects I implemented.
Current projects:
- Classification for a Telecom company's marketing campaign:
  - [EDA](https://github.com/phamthiminhtu/data_science_projects/blob/master/classification_marketing__campaigns_EDA.ipynb)
  - [Modelling](https://github.com/phamthiminhtu/data_science_projects/blob/master/classification__marketing_campaigns.ipynb):
      - Highlights: apply Bayesian Logistic Regression in predicting the binary target variable.
-------
A brief note about the Bayesian Methods.
Reference: this note is taken from Advanced Bayesian Methods course's materials by Professor Matt Wand, UTS.

# 1. Notation.

- $x$ is a random variable (instead of $X$)
- $p(x)$ is a density function of x at x (instead of )
- $p(x,y)$ is the joint density function of x and y
- $x = x^o$  (x ring or x circle) corresponds to $x$ having the observed value $x^o$.

# 2. Undirected graph and DAG.

## 2.1. Markov blanket.

- Markov blanket of a node in an UNDIRECTED graph is the set of all the neighbors of the node.
- Markov blanket of a node in a DAG is the set of **parents, co-parents and children** of that node.

## 2.2. Ancestral sub-graphs.

A sub-graph of a DAG is called an ancestral sub-graph if for any node in the sub-graph, each of that node’s ancestors is also in the sub-graph.

→ The ancestral sub graph that has the fewest number of node is called the **smallest ancestral sub-graph.**

## 2.3. Moral graph.

Moralization:

- Step 1: For each node, if any **pair of parents** of that node are not connected by an edge, then add an undirected edge between them.
- Step 2: Change all the directed edges to undirected edges.

Note: the Markov blanket of a node in a DAG is **identical** to the Markov blanket of that node in the DAG’s moral graph.

## 2.4. Clique and maximal clique.

- A clique is a subset of vertices of an undirected graph such that **every two distinct nodes** in the clique are connected (adjacent).
- A maximal clique is a clique that cannot be extended by including one more adjacent vertex, meaning it is not a subset of a larger clique. A maximum clique (i.e., clique of largest size in a given graph) is therefore always maximal, but the converse does not hold.

# 3. Probabilistic Undirected Graph.

## 3.1. Joint probability of nodes.

$$
p(x_1, x_2, ... , x_k) ∝ \psi(C_1)*\psi(C_2)*...*\psi(C_l)
$$



where:

- $\psi$ is the potential function.
- C_1 … C_l are the maximal cliques of the UNDIRECTED graph.

IMPORTANT!

Note that the density function is **proportional** to a product of potential functions, where the potential functions are defined over the maximal cliques.

# 4. Probabilistic DAG.

## 4.1. Joint probability of nodes.

$$
p(x_1, x_2, ... , x_k) = \Pi_{k=1}^k p(x_k| \text{ parents of }  x_k)
$$

# 5. Conditional independence theorems.

## 5.1. Probabilistic UNDIRECTED graph.

If two distinct nodes, $x_i$ and  $x_j$, in a probabilistic UNDIRECTED graph, are not connected by an edge then:

$$
x_i ⫫ x_j | rest
$$

Let A, B and C be three disjoint node subsets of a probabilistic UNDIRECTED graph. Then

$$
A ⫫ B | C
$$

if and only if C separates A from B.

## 5.2. Probabilistic DAG.

Let A, B and C be three disjoint node subsets of a probabilistic DAG. Then

$$
A ⫫ B | C
$$

if and only if C separates A from B in the **moral graph** of the SMALLEST ancestral sub-graph containing $A ∪ B ∪ C$

# 6. The locality property of probabilistic graphs.

## 6.1. Full conditional distribution.

$$
x_i | rest, 1< i < k
$$

is the full conditional distributions. The corresponding density function are:

$$
p(x_i | rest), 1< i < k
$$

## 6.2. The locality of UNDIRECTED graphs or DAG.

Let $p(x_1,...,x_k)$ be a probability density function defined on **either** an undirected graph or DAG. Then:

$$
x_i | rest \text{ has the same distribution as } x_i| \text{Markov blanket of } x_i, 1< i < k
$$

And the corresponding density functions are:

$$
p(x_i | rest) = p(x_i | \text{ Markov Blanket of } x_i), 1< i < k
$$

- In the case of UNDIRECTED graph: Markov blanket is all the neighbors in the graph.
- In the case of DAG: Markov blanket are:
    - Co-parents
    - Parents
    - Children

This property helps simplify the conditional density function in 6.1:

Example:

- In the case of probabilistic UNDIRECTED graph:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/d4f5836d-559b-44c2-b7d5-e9da452c9aab/Untitled.png)

$$
p(A| rest) = P(A| D, B)
$$

$$
p(C| rest) = p(C| D, B)
$$

and so on

- In the case of probabilistic DAG:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/af4159c4-fe1a-44a2-97dc-04abfc0556ae/Untitled.png)

$$
p(Z| rest) = p(Z|X, Y, W)
$$

$$
p(W| rest) = p(W| Z)
$$

and so on

# 7. Marginalization.

Example: Given the UNDIRECTED graph:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/db8a58fd-270b-45a7-96da-32f1428e2bc3/Untitled.png)

$$
p(x_1) = \Sigma_{x_2=0}^1\Sigma_{x_2=0}^1...\Sigma_{x_{13}=0}^1p(x_2,...,x_{13})

$$

Note  $x_1$ is not involved in the sums

## 7.1. Extension to conditional marginalization.

Conditional marginalization of a probabilistic graph involves obtaining the conditional density functions of the nodes when SOME of the NODES are **OBSERVED**.

Conditional marginalization applies for both UNDIRECTED graph and DAG. The core concept is :

$$
p(x_1 | x_2, x_3...x_{k}) = \frac{p(x_1, x_2, x_3, ..., x_k)}{p(x_2 , x_3,..., x_k)}
$$

If some of the nodes are observed, then just SUBSTITUTE their observed values into the equation (instead of integrate or sum over their all possible values).

### 7.1.1. **Example.**

- Let’s say $x_2 \text{ and } x_3 \text{ are observed, where } x_2 = 0, x_3 = 1$.
- $x_1 = 0, 1, x_4 = 0,1$
- $p(x_1, x_2=0, x_3=1, ..., x_k) = 2x_1 + 3x_2 + 4x_3 + 5x_4$

Then the equation becomes:

$$
p(x_1 | x_2=0, x_3=1...x_{k}) = \frac{p(x_1, x_2=0, x_3=1, ..., x_k)}{p(x_2=0 , x_3=1,..., x_k)}
$$

$$
= \frac{\sum_{x_4=0}^12x_1 + 3*0 + 4*1 + 5x_4}{\sum_{x_1=0}^1\sum_{x_4=0}^12x_1 + 3*0 + 4*1 + 5x_4}
$$

### 7.1.2. **Another concrete example.**

Given the DAG:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/a72af071-ba6e-46ea-8e5a-82d07acfa339/Untitled.png)

Note that

$$
p(x_4| x_1=0, x_2=1, x_3=1, x_8=0) ∝ p(x_4, x_1=0, x_2=1, x_3=1, x_8=0)
$$

$$
= \sum_{x_5=0}^1\sum_{x_6=0}^1\sum_{x_7=0}^1\sum_{x_9=0}^1\sum_{x_{10}=0}^1\sum_{x_{11}=0}^1p(x_4, x_1=0, x_2=1, x_3=1, x_5, x_6, x_7, x_8=0, x_9, x_{10}, x_{11})
$$

Note that: proportional in 

$$
p(x_4| x_1=0, x_2=1, x_3=1, x_8=0) ∝ p(x_4, x_1=0, x_2=1, x_3=1, x_8=0)
$$

because the denominate is the constant:

$$
p(x_1, x_2, x_3, x_8) = \sum_{x_4=0}^1\sum_{x_5=0}^1\sum_{x_6=0}^1\sum_{x_7=0}^1\sum_{x_9=0}^1\sum_{x_{10}=0}^1\sum_{x_{11}=0}^1 p(x_1=0, x_2=1, x_3=1, x_4, x_5, x_6, x_7, x_8=0, x_9, x_{10}, x_{11})
$$

 which DOES NOT depend on $x_4$.

### 7.1.3. E**xample on continuous variables.**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/c2b9b6e1-ffb6-4112-989c-ed93e9a882eb/Untitled.png)

## 7.2. Conditional marginalization in probabilistic graph with conditional independence theorems.

Example: given the DAG and its moral graph.

**DAG 7.2.1:**

![IMG_6176.jpeg](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/f6c7b658-231f-4eaa-9fce-695bd9cfe770/IMG_6176.jpeg)

And:

$$
p(x_1, x_2, x_3, x_4) ∝ exp(2x_1 + 3x_2 + 4x_3 + 5(x_4 - x_2)^2)
$$

We have

$$
x_1 ⫫ x_4 | x_2, x_3
$$

So:

$$
p(x_1, x_4| x_2, x_3) = p(x_1|x_2, x_3) * p(x_4|x_2, x_3)
$$

Then 

$$
p(x_1| x_2, x_3)  = \frac{p(x_1, x_2, x_3)}{p(x_2, x_3)} 
$$

$$
∝ p(x_1, x_2, x_3) = \int{p(x_1, x_2, x_3, x_4)}dx_4
$$

$$
= \int exp(2x_1 + 3x_2 + 4x_3 + 5(x_4 - x_2)^2)dx_4
$$

$$
= exp(2x_1 + 3x_2 + 4x_3)\int exp(5(x_4 - x_2)^2) dx_4
$$

$$
∝ exp(2x_1 + 3x_2 + 4x_3)
$$

because $\int exp(5(x_4 - x_2)^2) dx_4$ is the constant which does not depend on $x_1$

Another example using the same DAG above:

![IMG_6177.jpeg](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/6fb1bba6-64a1-4465-9029-10922f2cc6ae/IMG_6177.jpeg)

# 8. Markov chain Monte Carlo.

## 8.1. How to MCMC.

![IMG_6179.jpeg](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/8e1d78d6-67e3-41ba-b621-01b6a4422ec6/IMG_6179.jpeg)

Example:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/e64973dd-d863-4085-bc63-f6a30f93de5a/Untitled.png)

Our goals are to obtain the conditional marginal distributions:

$$
p(x_1|x_2^o) \text{ and }p(x_3|x_2^o)
$$

But the analytical way is intractable because of the complicated integrations without a close form solution 

⇒ use MCMC.

Solution:

First, we have:

$$
p(x_1| x_2, x_3)∝ p(x_1, x_2, x_3) ∝ p(x_1)p(x_2|x_1, x_3)p(x_3)
$$

$$
∝ p(x_1)p(x_2|x_1, x_3)
$$

$$
∝ exp(-\frac{1}{2}x_1^2)exp(-\frac{1}{2}x_3(x_2 - x_1)^2)
$$

This leads to:

$$
-2log(p(x_1| x_2, x_3)) = x_3(x_2 - x_1)^2) + x_1^2 = x_3(x_2^2 - 2x_2x_1 + x_1^2) + x_1^2
$$

$$
= (x_3 + 1)x_1^2 - 2x_1x_2x_3 + x_2^2x_3 = (x_3 + 1)x_1^2 - 2x_1x_2x_3 + const
$$

where const terms do not depend on $x_1$.

**(1)** Then $x_1$ has a close form distribution: 

$$
x_1| x_2^o, x_3 \text{\~
} N(\frac{x_2^ox_3}{x_3+1}, \frac{1}{x_3+1})
$$

Similarly, we have:

$$
p(x_3| x_2, x_1)∝ p(x_1, x_2, x_3) ∝ p(x_1)p(x_2|x_1, x_3)p(x_3)
$$

$$
∝ p(x_3)p(x_2|x_1, x_3)
$$

Note that:

$$
p(x_2|x_1, x_3) = \frac{1}{\sqrt{(2\pi \frac{1}{x_3})}}exp(\frac{-(x_2-x_1)^2}{\frac{2}{x_3}})
$$

$$
∝ x_3^{\frac{1}{2}}exp(-\frac{1}{2}x_3(x_2 - x_1)^2)
$$

$$
= x_3^{\frac{3}{2} - 1}exp(-\frac{1}{2}x_3(x_2 - x_1)^2)
$$

and:

$$
p(x_3) = 1/1x_3^0exp(-x_3) = exp(-x_3)
$$

Thus we have:

$$
p(x_3| x_2, x_1)∝ x_3^{\frac{3}{2} - 1}exp(-x_3)exp(-\frac{1}{2}x_3(x_2 - x_1)^2)
$$

$$
= x_3^{\frac{3}{2} - 1}exp(-x_3(\frac{1}{2}(x_2-x_1)^2 + 1))
$$

**(2)** Then $x_3$ has a close form distribution: 

$$
x_3| x_2^o, x_1 \text{ \~
 } Gamma(\frac{3}{2}, \frac{1}{2}(x_2^o -x_1)^2+1)
$$

From (1) and (2), we have the MCMC scheme as follow:

- Initialize $x_3^{[0]}$
- Cycle: $g = 1..., B+K$
    - $x_1^{[g]} \text{ \~
     } N(\frac{x_2^ox_3^{[g-1]}}{x_3^{[g-1]}+1}, \frac{1}{x_3^{[g-1]}+1})$
    - $x_3^{g} \text{ \~
     } Gamma(\frac{3}{2}, \frac{1}{2}(x_2^o -x_1^g)^2+1)$

## 8.2. Relationship to locality property.

MCMC requires the full conditional probability function.

But thanks to the locality property, we can simplify it to the conditional probability function on the Markov blanket instead of “rest” .

![IMG_6181.jpeg](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/f746782d-09a2-4aaa-bb49-a36fba227b42/IMG_6181.jpeg)

# 9. MCMC in relation to multiple data points.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/ff1a7c97-8c6e-4c8b-a686-24180bc5279c/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/08a6a05c-5ee9-451a-852d-c3db9377835b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/cfc305bf-63a1-41c5-8b30-1e3833d7d1cb/Untitled.png)

**Note:**

- We have

$$
p(\bold y|\beta_1, \beta_0, \sigma^2) = \prod_{i=1}^np(y_i|\beta_1, \beta_0, \sigma^2)
$$

because the data points are **independent**.

- And we have  $p(\bold y|\beta_1, \beta_0, \sigma^2)$ equals to:

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/84a4a58a-5bca-473e-9f1c-4770927d73d7/Untitled.png)

because 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/34e88326-2db0-45f2-8a39-69cbcb130b4f/424d42eb-7fad-4a84-9460-6c586bd80be0/Untitled.png)

i.e: $y_i$ follows Normal distribution with mean $\beta_0 + \beta_1x_i$ and variance $\sigma^2$.

