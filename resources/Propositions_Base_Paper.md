### - The serial supply chain system is Partially Observable Markov Game

---

### Why does the observation include historical/in-transit information?

$$ L=4 $$

Then an order/shipment doesn't immediately become available.

There is a four-period delay.

Therefore, the actor needs information about what has been shipped in previous periods to understand what inventory will arrive in the future.

This is one reason the authors use a GRU recurrent neural network.

The GRU can maintain information from previous observations rather than making a decision solely from the current observation.

---

### Supply-chain dynamics inside the POMG

This is important because the POMG isn't an abstract mathematical game—the transition function comes from the actual inventory system.

For actor \(i\):

$$ P_t^i = \min \left( B_t^i+O_t^{i-1}, I_t^i+P_{t-L}^{i} \right) $$

Meaning:

Demand that needs to be fulfilled
$$ B_t^i+O_t^{i-1} $$

Existing backlog + new downstream order.

Inventory available to fulfill it
$$ I_t^i+P_{t-L}^{i} $$

Current inventory + shipment that has just arrived after the lead time.

Therefore:

$$ \text{shipment} = \min(\text{demand},\text{available inventory}) $$
9. Backlog transition

The backlog evolves according to:

$$ B_{t+1}^i = B_t^i+O_t^{i-1}-P_t^i $$

Think of this as:

$$ \boxed{ \text{New backlog} = \text{old backlog} + \text{new demand} - \text{fulfilled demand} } $$

If demand is greater than what can be shipped, backlog increases.

10. Inventory transition

Inventory evolves according to:

$$ I_{t+1}^i = I_t^i+P_{t-L}^i-P_t^i $$

Meaning:

$$ \boxed{ \text{New inventory} = \text{current inventory} + \text{arrivals} - \text{shipments} } $$

For your \(L=4\):

$$ I_{t+1}^i = I_t^i+P_{t-4}^i-P_t^i $$
11. Cost function

Each actor has:

$$ C_t^i=h_iI_t^i+b_iB_t^i $$

For your setup:

$$ h_i=1,\qquad b_i=1 $$

so:

$$ C_t^i=I_t^i+B_t^i $$

The total supply-chain cost is:

$$ C_t^{total} = \sum_{i=1}^{M}C_t^i $$

For four echelons:

$$ C_t^{total} = C_t^1+C_t^2+C_t^3+C_t^4 $$
12. The fundamental optimization objective

The supply chain is fully cooperative.

Therefore, the fundamental objective is to minimize:

$$ \mathbb E \left[ \sum_{t=0}^{\infty} \gamma^t C_t^{total} \right] $$

where \(\gamma\) is the discount factor.

The paper uses:

$$ \gamma=0.95 $$

So the system wants policies that minimize the long-term discounted total cost of the entire supply chain, not merely the cost of one echelon.

---

### why doesn't every agent simply minimize total cost?

- Because the author found that HAPPO can perform better when agents care partly about their own costs as well.
- For the 4-echelon serial setup, the base paper later selects:

$$ \boxed{\alpha=0.375} $$

so:

$$ r_t^i = -0.375C_t^i - 0.625C_t^{total} $$

Again, this value belongs to the 4-echelon serial-supply-chain experiment.

---

# Why introduce \(\alpha\) at all?

This is the central theoretical idea of the paper.

The authors argue that \(\alpha\) creates a bias–variance trade-off in MADRL training.

There are two competing effects.

Effect 1 — Lower α

As α decreases:

The reward becomes a better representation of the true system objective.

Therefore, the reward has less bias relative to:

$$ -C_t^{total} $$

This is good.

But...

Effect 2 — Lower α

The reward becomes more dependent on the total cost of all the other actors.

That causes the accumulated reward to have higher variance, making it harder to estimate accurately from a finite number of simulated episodes.

This is bad.

---

# Proposition 1 — Bias

The base paper states, in essence:

When a joint policy is applied, the bias between an actor's expected accumulated reward and the negative expected accumulated total supply-chain cost is monotonically increasing with \(\alpha\), and the bias is zero when \(\alpha=0\).

Let's unpack this carefully.

#### What is the "true" objective?

The true system objective is:

$$ -\mathbb E \left[ \sum_t\gamma^t C_t^{total} \right] $$

because maximizing negative cost is equivalent to minimizing cost.

But the actor isn't necessarily maximizing that directly.

It maximizes:

$$ \mathbb E \left[ \sum_t\gamma^t r_t^i \right] $$

where:

$$ r_t^i = -\alpha C_t^i - (1-\alpha)C_t^{total} $$
#### Substitute the reward

The accumulated reward is:

$$ \sum_t\gamma^t r_t^i = -\alpha \sum_t\gamma^t C_t^i - (1-\alpha) \sum_t\gamma^t C_t^{total} $$

Compare this with the desired quantity:

$$ -\sum_t\gamma^t C_t^{total} $$

They are identical only when:

$$ \alpha=0 $$

#### What happens at \(\alpha=0\)?

Set:

$$ \alpha=0 $$

Then:

$$ r_t^i = -C_t^{total} $$

Therefore:

$$ \sum_t\gamma^t r_t^i = -\sum_t\gamma^t C_t^{total} $$

Exactly.

So the actor's reward perfectly represents the system-level objective.

Therefore:

$$ \boxed{\text{Bias}=0} $$
19. What happens when \(\alpha>0\)?

Suppose:

$$ \alpha=0.375 $$

Then:

$$ r_t^i = -0.375C_t^i - 0.625C_t^{total} $$

The actor is no longer optimizing exactly the total system cost.

It is optimizing a mixture of:

$$ \text{own cost} $$

and:

$$ \text{system cost} $$

Therefore, the optimization target has shifted away from the pure system objective.

That creates bias.

#### Why does Proposition 1 say bias increases with α?

Because increasing \(\alpha\) gives increasing weight to:

$$ -C_t^i $$

and decreasing weight to:

$$ -C_t^{total} $$

So the actor's reward progressively becomes less representative of the true common objective.

At:

$$ \alpha=0 $$

there is no deviation.

At:

$$ \alpha=1 $$

the actor is optimizing only its own cost.

Thus, conceptually:

α = 0
│
│  exact system objective
│
├───────────────→ α
│
│ increasingly individual objective
│
α = 1

Hence:

$$ \boxed{ \alpha\uparrow \Rightarrow \text{bias}\uparrow } $$

---

# Proposition 2 — Variance

Now comes the other side of the trade-off.

The paper states that, under the condition that the relevant costs of actor \(i\) and the other actors are positively correlated, the variance of actor \(i\)'s accumulated reward is monotonically decreasing with \(\alpha\).

This sounds counterintuitive at first.

Let's unpack it.

#### What causes variance?

HAPPO doesn't have access to the true expected reward analytically.

Instead, it estimates expected rewards by simulating a finite number of episodes.

The paper explains that before each policy update, the current joint policy is evaluated by running several simulated episodes and using the resulting rewards to estimate expected reward.

Therefore:

$$ \text{finite simulations} \rightarrow \text{sampling uncertainty} \rightarrow \text{variance} $$
#### Write the accumulated reward

For actor \(i\):

$$ R_i = \sum_t\gamma^t r_t^i $$

Substitute the reward:

$$ R_i = -\alpha \sum_t\gamma^t C_t^i - (1-\alpha) \sum_t\gamma^t C_t^{total} $$

Now remember:

$$ C_t^{total} = C_t^i+ \sum_{j\ne i}C_t^j $$

Therefore the actor's accumulated reward depends on:

its own cost
costs of all the other actors
#### Why does lower α increase variance?

When:

$$ \alpha\rightarrow0 $$

the reward becomes:

$$ R_i = -\sum_t\gamma^t C_t^{total} $$

That means actor \(i\)'s reward depends on the entire supply chain's cost.

So fluctuations in:

actor 1's cost
actor 2's cost
actor 3's cost
actor 4's cost

can all influence actor \(i\)'s reward.

Consequently, there can be substantial variation in the reward from one simulation episode to another.

#### Compare with α = 1

At:

$$ \alpha=1 $$

we have:

$$ r_t^i=-C_t^i $$

Therefore:

$$ R_i = -\sum_t\gamma^tC_t^i $$

Now the reward only depends on actor \(i\)'s own cost.

There are fewer sources of randomness contributing to its reward.

Thus the estimated expected reward can have lower variance.

#### The role of positive correlation

The paper's Proposition 2 has an important condition.

It assumes that the cost of actor \(i\) has a non-negative covariance with the costs of the other actors.

Conceptually:

$$ Cov(C_i,C_j)\ge0 $$

for the relevant accumulated discounted costs.

Why is this important?

Because in a supply chain, actors' costs tend to move together.

For example:

Demand spike
     ↓
Echelon 1 backlog ↑
     ↓
Echelon 2 pressure ↑
     ↓
Echelon 3 inventory/backlog changes
     ↓
Echelon 4 inventory changes

So high cost at one actor tends to be associated with high costs elsewhere.

Under this positive-correlation condition, including more of the total-system cost in an actor's reward increases the reward's variance.

Therefore:

$$ \boxed{ \alpha\downarrow \Rightarrow \text{variance}\uparrow } $$

or equivalently:

$$ \boxed{ \alpha\uparrow \Rightarrow \text{variance}\downarrow } $$

---

# Proposition 3

- Proposition 3 is actually very important, because unlike Propositions 1 and 2—which explain the bias–variance trade-off caused by α—Proposition 3 explains a theoretical convergence/improvement property of HAPPO itself.

#### First, what is Proposition 3 saying?

In simple terms:

> **If we use HAPPO on the serial supply chain and set α = 0, every sequential policy update made by HAPPO will not increase the expected long-term total supply-chain cost.**

In other words:

$$
\boxed{
\text{After a HAPPO policy update: expected total cost decreases or stays the same}
}
$$

So if the policy at iteration \(k\) is

$$
(\pi_1^k,\pi_2^k,\ldots,\pi_M^k)
$$

and after the HAPPO update we have

$$
(\pi_1^{k+1},\pi_2^{k+1},\ldots,\pi_M^{k+1}),
$$

then the paper claims:

$$
\boxed{
J(\pi^{k+1})\leq J(\pi^k)
}
$$

where \(J(\pi)\) is the expected discounted **overall supply-chain cost**.

---

#### Why does α = 0 matter?

Remember the reward function:

$$
r_t^i
=
-\alpha C_t^i
-
(1-\alpha)C_t^{total}
$$

where:

* \(C_t^i\) = cost of actor/echelon \(i\)
* \(C_t^{total}\) = total cost of the whole supply chain
* \(\alpha\) = degree of self-interest.

At:

$$
\alpha=0
$$

we get:

$$
r_t^i
=
-C_t^{total}
$$

for **every actor**.

So all actors are literally trying to maximize:

$$
-C_t^{total}
$$

which is equivalent to minimizing:

$$
C_t^{total}.
$$

Therefore, at \(\alpha=0\):

$$
\boxed{
\text{Every actor has exactly the same optimization objective.}
}
$$

This is why Proposition 3 is formulated specifically for \(\alpha=0\). 

---

#### What exactly is the equation in Proposition 3?

The paper writes the proposition approximately as:

$$
E_{\pi^{1,k+1},\ldots,\pi^{M,k+1}}
\left[
\sum_{t=0}^{\infty}
\gamma^t C_t^{total}
\right]
\leq
E_{\pi^{1,k},\ldots,\pi^{M,k}}
\left[
\sum_{t=0}^{\infty}
\gamma^t C_t^{total}
\right].
$$



Let's break this down.

---

## Left side

$$
E_{\pi^{1,k+1},\ldots,\pi^{M,k+1}}
\left[
\sum_{t=0}^{\infty}
\gamma^t C_t^{total}
\right]
$$

means:

> **Expected discounted total cost when all actors use the newly updated policies.**

The superscript \(k+1\) means:

> HAPPO has completed another policy-update iteration.

---

## Right side

$$
E_{\pi^{1,k},\ldots,\pi^{M,k}}
\left[
\sum_{t=0}^{\infty}
\gamma^t C_t^{total}
\right]
$$

means:

> **Expected discounted total cost under the previous policies.**

So Proposition 3 simply says:

$$
\boxed{
\text{New expected cost}
\leq
\text{Old expected cost}
}
$$

---

#### Why does HAPPO guarantee this?

This is where **HAPPO's sequential policy update** becomes important.

HAPPO does **not** update all actors simultaneously.

Instead, it updates the actors sequentially.

Suppose we have four echelons:

$$
\pi_1,\pi_2,\pi_3,\pi_4
$$

HAPPO can conceptually perform:

$$
\pi_1^k
\rightarrow
\pi_1^{k+1}
$$

then

$$
\pi_2^k
\rightarrow
\pi_2^{k+1}
$$

then

$$
\pi_3^k
\rightarrow
\pi_3^{k+1}
$$

then

$$
\pi_4^k
\rightarrow
\pi_4^{k+1}.
$$

The important point is that **each actor's update is evaluated while accounting for the policies of the other actors**.

The paper describes HAPPO as using a **sequential policy update scheme**, and Proposition 3 gives the resulting monotonic improvement guarantee for the fully cooperative case.

---

#### Think of one HAPPO iteration

Suppose initially:

$$
(\pi_1^k,\pi_2^k,\pi_3^k,\pi_4^k)
$$

produces an expected total cost of:

$$
J_k=10,000.
$$

Now HAPPO updates actor 1.

After the update:

$$
(\pi_1^{k+1},\pi_2^k,\pi_3^k,\pi_4^k)
$$

and the expected cost should satisfy:

$$
J_1\leq10,000.
$$

Suppose:

$$
J_1=9,700.
$$

Then HAPPO updates actor 2:

$$
(\pi_1^{k+1},\pi_2^{k+1},\pi_3^k,\pi_4^k)
$$

and:

$$
J_2\leq9,700.
$$

Maybe:

$$
J_2=9,400.
$$

Then actor 3:

$$
J_3\leq9,400.
$$

Then actor 4:

$$
J_4\leq9,400.
$$

Maybe:

$$
J_4=9,100.
$$

Therefore, one complete HAPPO iteration could look like:

$$
10,000
\rightarrow
9,700
\rightarrow
9,400
\rightarrow
9,400
\rightarrow
9,100.
$$

So the sequence is **non-increasing**.

That's what "monotonically decreases" means.

---

#### Why is this particularly useful for a supply chain?

This is important because the supply chain is **multi-agent**.

Imagine four autonomous echelons:

```text
Customer
   ↓
Echelon 1
   ↓
Echelon 2
   ↓
Echelon 3
   ↓
Echelon 4
   ↓
Manufacturer
```

Each actor makes its own ordering decision.

The problem is that changing one actor's policy can affect:

* downstream inventory,
* upstream orders,
* backlogs,
* shipments,
* inventory holding,
* and ultimately the total supply-chain cost.

So naively updating all policies at once can make the learning process unstable.

HAPPO instead performs **controlled sequential updates**.

That is one of the theoretical reasons the paper chooses HAPPO for this problem.

---

#### The role of the critic

During training, HAPPO uses information about the overall system through the critic.

For the full-observation training setting, the critic can use the observation histories of all actors.

Conceptually:

```text
Actor 1 observation ──┐
Actor 2 observation ──┤
Actor 3 observation ──┼──> Centralized Critic
Actor 4 observation ──┘
                            │
                            ↓
                       Value estimate
                            │
                            ↓
                     Policy update
```

The actor itself still learns its decentralized ordering policy from its own observation history.

So HAPPO uses the centralized information to make the **training/update process** more informed, while the resulting actors can operate using local information.

The paper explicitly describes this CTDE structure for the POMG formulation. 

---

#### But there is an extremely important distinction

You might now ask:

> "If Proposition 3 says α = 0 gives monotonic improvement, why did the paper find α = 0.375 to perform better for my 4-echelon system?"

This is **the key point**.

There is no contradiction.

---

- Proposition 3 is a theoretical guarantee

It says:

$$
\boxed{\alpha=0}
$$

gives a theoretically aligned objective, and under HAPPO's sequential policy update scheme, expected overall costs monotonically improve.

But this does **not** mean:

$$
\boxed{\alpha=0\text{ will produce the best practical trained model}}
$$

because the actual MADRL algorithm uses **finite simulations to estimate expected rewards**.

That's where Proposition 2 comes in.

---

#### Put Propositions 1, 2 and 3 together

This gives you the full theoretical story of the paper.

- Proposition 1 — Bias

As α increases:

$$
\alpha\uparrow
\Rightarrow
\text{bias}\uparrow
$$

because the actor's objective becomes increasingly different from the true system objective.

At:

$$
\alpha=0
$$

bias is zero.

---

- Proposition 2 — Variance

Under the paper's covariance condition:

$$
\alpha\uparrow
\Rightarrow
\text{reward-estimation variance}\downarrow.
$$

So larger α makes the finite-simulation estimate more statistically stable.



---

- Proposition 3 — Monotonic policy improvement

At:

$$
\alpha=0
$$

the objective is exactly the common system objective, and HAPPO's sequential update scheme guarantees:

$$
J(\pi^{k+1})\leq J(\pi^k).
$$

---

#### So why not simply use α = 0?

Because there are **two different levels of analysis**.

### Theoretical level

At α=0:

```text
Perfect objective alignment
          ↓
No bias
          ↓
HAPPO sequential improvement guarantee
```

Excellent.

- Practical training level

At α=0:

```text
Reward = negative total system cost
          ↓
Reward contains cost fluctuations
          ↓
Higher variance in finite-sample estimation
          ↓
Noisier policy evaluation
          ↓
Potentially less effective learning
```

And this is exactly why the paper introduces the α bias–variance trade-off.

The paper's experiments find that the best practical α is **between 0 and 1**, rather than exactly 0. 

For the **4-echelon serial supply chain**, they empirically select:

$$
\boxed{\alpha=0.375}
$$

because that value produced the lowest overall cost in their α-sensitivity experiment. 

---

# 12. Very important: what Proposition 3 does NOT say

Don't interpret Proposition 3 as:

> "HAPPO will always find the globally optimal supply-chain policy."

That is **too strong**.

It says that under the stated theoretical setting—particularly the **serial supply chain with α=0**—the expected overall cost **monotonically improves with the sequential policy updates**.

That is different from saying:

$$
\text{HAPPO finds global optimum}.
$$

Monotonic improvement means:

$$
J_{k+1}\leq J_k
$$

not necessarily:

$$
J_k=J^*.
$$ 

---

# Proposition 4 — Optimality of HAPPO

Proposition 4 establishes a **known optimal policy** for a simplified serial supply-chain problem and then uses it as a benchmark to test whether HAPPO can learn the optimum. 

The proposition considers a serial supply chain with \(M\) echelons under three conditions:

1. **Zero lead time:** \(L=0\)
2. **No initial inventory:** \(I_0^i=0\) for every echelon
3. **No initial backlog:** \(B_0^i=0\) for every echelon

With these conditions, if customer demand \(O_t^0\) is **deterministic and known**, the optimal policy is simply:

$$
\boxed{O_t^i=O_t^0,\qquad \forall i=1,\ldots,M,\;t\geq0}
$$

In words:

> **Every echelon should order exactly the amount demanded by the customer in that period.**

Because there is **no lead time**, **no starting inventory**, and **no starting backlog**, there is no reason to order more or less than the known demand. Ordering exactly the demand avoids both unnecessary holding inventory and backlog costs. The paper proves this result in Appendix A and then trains HAPPO in this simplified environment to see whether it converges to this known optimal policy. 

But the authors **also test a second, more difficult optimality setting**:

* \(L>0\) (non-zero lead time)
* no startup inventory
* no startup backlog
* stochastic customer demand:

  $$
  O_t^0\sim\text{i.i.d. Poisson}
  $$
* backlog cost exists **only at the most downstream actor**:

  $$
  b^1=1,\qquad b^i=0\;(i=2,\ldots,M)
  $$

For this second setting, the paper relies on the established result that the **optimal policy is a base-stock policy**. It computes the optimal base-stock levels using the method of **Shang and Song (2003)** and then checks whether HAPPO can converge toward those known optimal levels.

The numerical results show that:

* In the **deterministic case**, HAPPO's ordering, inventory, and cost behavior converges toward the known optimal policy.
* In the **stochastic case**, HAPPO's learned base-stock levels gradually stabilize toward the optimal values.
* The overall-cost optimality gap falls from **more than 1500% initially to around 15% after training**.
* Therefore, under both simplified settings, HAPPO **almost closes the gap with the known optimal policies**, providing evidence that HAPPO can effectively learn near-optimal policies for multi-echelon inventory problems. 

### The key distinction

$$
\boxed{\text{Proposition 3: HAPPO keeps improving the expected cost}}
$$

$$
\boxed{\text{Proposition 4: HAPPO can approach a known optimal policy}}
$$

So **Proposition 3 is about monotonic improvement**, whereas **Proposition 4 is about convergence toward an actual optimum**, tested in two simplified serial-supply-chain settings.

