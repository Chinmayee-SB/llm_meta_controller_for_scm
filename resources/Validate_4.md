# 1. Validate and check if each of this is implemented in the project

what happens at \(t=0\) of a training episode?

For your target experiment:

$$ \boxed{M=4,\quad L=4,\quad \text{serial},\quad \text{Merton demand}} $$

there are 4 learning agents:

$$ A_1,A_2,A_3,A_4 $$

and the customer is actor 0, while the manufacturer is actor 5.

At the beginning of an episode, the environment has a global state containing the inventory/backlog/in-transit situation of the supply chain.

However, there is an important distinction:

The paper does not explicitly specify the exact initial inventory, initial backlog, and initial in-transit shipment values for the main \(M=4,L=4\) Merton experiment.

The paper explicitly specifies zero startup inventory and zero startup backlog only for the simplified optimality experiments of Proposition 4, not as a general initialization statement for the main experiment.

So we should not automatically say:

$$ I_0^1=I_0^2=I_0^3=I_0^4=0 $$

for the main experiment unless we have their implementation/code to verify it.

---

# 2. Validate and check if each of this is implemented in the project

So at \(t=0\), what does Agent 1 see?

Conceptually:

$$ o_0^1= (I_0^1,B_0^1,O_{-1}^{0}, P_{-4}^1,P_{-3}^1,P_{-2}^1,P_{-1}^1) $$

Agent 1 is the most downstream echelon, so its downstream actor is the customer.

It therefore needs:

its own initial inventory;
its own initial backlog;
previous customer order;
its four-period shipment history.

But there is a boundary problem:

$$ O_{-1}^{0},P_{-1}^1,P_{-2}^1,P_{-3}^1,P_{-4}^1 $$

don't naturally exist at the beginning of the simulation.

The paper does NOT tell us how those pre-\(t=0\) values are initialized.

It could be zero-padding, an environment-specific initialization, or some other convention—but we shouldn't claim one without the implementation.

This is one of those details where the paper's mathematical formulation is more precise than its implementation description.

Agent 2 at \(t=0\)

Agent 2 sees:

$$ o_0^2= (I_0^2,B_0^2,O_{-1}^{1}, P_{-4}^2,P_{-3}^2,P_{-2}^2,P_{-1}^2) $$

The important difference is:

$$ O_{-1}^{1} $$

instead of customer demand.

Agent 2 cares about the previous order from Echelon 1.

Agent 3

Similarly:

$$ o_0^3= (I_0^3,B_0^3,O_{-1}^{2}, P_{-4}^3,P_{-3}^3,P_{-2}^3,P_{-1}^3) $$
Agent 4

And:

$$ o_0^4= (I_0^4,B_0^4,O_{-1}^{3}, P_{-4}^4,P_{-3}^4,P_{-2}^4,P_{-1}^4) $$

Agent 4 is the most upstream intermediate echelon.

Its upstream supplier is the manufacturer.

The manufacturer has infinite stock, so Agent 4's ordering is unconstrained by manufacturer inventory.

---

# 3. Validate and check if each of this is implemented in the project

what does the critic see at \(t=0\)?

This is where CTDE comes in.

The actors don't see the entire supply chain.

For actor \(i\):

$$ \pi^i $$

uses its own observation history:

$$ \bar{o}_t^i=(o_0^i,\ldots,o_t^i). $$

But the critic for actor \(i\):

$$ V^i $$

maps:

$$ \boxed{ (\bar{o}_t^1,\bar{o}_t^2,\bar{o}_t^3,\bar{o}_t^4) } $$

to a value estimate.

The paper explicitly states that the value function uses all actors' observation histories, while the policy function uses the individual actor's partial observation history.

Therefore, at \(t=0\):

$$ \boxed{ V_0^i= V^i(o_0^1,o_0^2,o_0^3,o_0^4) } $$

because there is no previous history yet.

---

# 4. Validate and check if each of this is implemented in the project

let's look at TWO adjacent episodes

This is probably the most useful way to understand training.

Suppose:

$$ \boxed{\text{Episode }k} $$

is currently being trained.

At the beginning:

$$ \pi_1^k,\pi_2^k,\pi_3^k,\pi_4^k $$

are the current actor policies.

And:

$$ V_1^k,V_2^k,V_3^k,V_4^k $$

are the current critics.

The paper explicitly uses \(\pi^{i,k}\) and \(V^{i,k}\) to represent the networks after \(k\) updates.

Episode \(k\): start

At \(t=0\):

Current policies:

π¹ᵏ
π²ᵏ
π³ᵏ
π⁴ᵏ

The environment is reset to a new episode.

The actors receive:

$$ o_0^1,o_0^2,o_0^3,o_0^4. $$

The critics receive:

$$ (o_0^1,o_0^2,o_0^3,o_0^4). $$

Episode \(k\): first action

Each actor's GRU processes its observation:

$$ (o_0^i,e_0^i) \rightarrow GRU \rightarrow \text{features}. $$

Actor network then produces:

$$ \pi^{i,k}(a_0^i|o_0^i). $$

Each actor samples an order:

$$ a_0^i=O_0^i. $$

So the joint action is:

$$ \boxed{ \mathbf a_0= (O_0^1,O_0^2,O_0^3,O_0^4) } $$

The environment then executes these actions.

The supply chain responds

The environment calculates:

shipments;
inventory;
backlog;
goods in transit.

The shipment equation is:

$$ P_t^i= \min \left( B_t^i+O_t^{i-1}, I_t^i+P_{t-L}^i \right). $$

With \(L=4\):

$$ P_t^i= \min \left( B_t^i+O_t^{i-1}, I_t^i+P_{t-4}^i \right). $$

Then:

$$ I_{t+1}^i = I_t^i+P_{t-4}^i-P_t^i $$

and:

$$ B_{t+1}^i = B_t^i+O_t^{i-1}-P_t^i. $$
The rewards are generated

For your chosen:

$$ \alpha=0.375 $$

the reward is:

$$ r_t^i = -0.375C_t^i - 0.625C_t^{total}. $$

The experience at time \(t\) therefore contains things like:

Agent i:

observation
    ↓
action
    ↓
reward
    ↓
next observation
    ↓
next hidden state

The paper says that observations, actions and rewards are collected as training data over the finite episode.

This continues until \(T=200\)

The paper specifies:

$$ \boxed{T=200} $$

periods per episode.

So:

Episode k

t=0
 ↓
t=1
 ↓
t=2
 ↓
...
 ↓
t=198
 ↓
t=199
 ↓
t=200

The exact indexing convention around whether the 200-period trajectory is represented as \(0,\ldots,199\) or includes a terminal \(t=200\) state depends on implementation; the paper simply reports episode length \(T=200\).

Then the important thing happens: POLICY UPDATE

After collecting the episode's data, HAPPO updates the networks.

The paper says:

for each episode, simulate the POMG with the actors' current policies and collect all actors' observations, actions, and rewards; then the actor and critic networks are updated using the collected data.

So conceptually:

$$ \boxed{ \text{Episode }k \rightarrow \text{collect trajectory} \rightarrow \text{HAPPO update} \rightarrow \text{Episode }k+1 } $$
Episode \(k+1\) is therefore different

This is the critical point.

At the beginning of the next episode:

$$ \boxed{ \pi^{1,k+1},\pi^{2,k+1},\pi^{3,k+1},\pi^{4,k+1} } $$

are now being used.

So the next episode does not use exactly the same policies as episode \(k\).

Conceptually:

                 EPISODE k
                     │
       π¹ᵏ π²ᵏ π³ᵏ π⁴ᵏ
                     │
                     ▼
              200-period rollout
                     │
                     ▼
              Training dataset
                     │
                     ▼
                HAPPO UPDATE
                     │
                     ▼
       π¹ᵏ⁺¹ π²ᵏ⁺¹ π³ᵏ⁺¹ π⁴ᵏ⁺¹
                     │
                     ▼
                 EPISODE k+1

This is the most important difference between adjacent training episodes.

Another difference between Episode \(k\) and Episode \(k+1\)

The environment trajectory will also generally be different.

Because customer demand is stochastic:

$$ O_t^0= \left\lfloor10e^{J_t}\right\rfloor $$

and the Merton process contains random variables:

$$ Z_t,\quad Z'_t,\quad N_t. $$

So even if you hypothetically kept the policies identical, a new episode could produce a different demand trajectory.

Therefore:

$$ \boxed{ \text{Episode }k\neq\text{Episode }k+1 } $$

for two reasons:

Reason 1 — policy changed
$$ \pi^k\rightarrow\pi^{k+1} $$
Reason 2 — stochastic environment
$$ D^k(t)\neq D^{k+1}(t) $$

in general.another difference between Episode \(k\) and Episode \(k+1\)

The environment trajectory will also generally be different.

Because customer demand is stochastic:

$$ O_t^0= \left\lfloor10e^{J_t}\right\rfloor $$

and the Merton process contains random variables:

$$ Z_t,\quad Z'_t,\quad N_t. $$

So even if you hypothetically kept the policies identical, a new episode could produce a different demand trajectory.

Therefore:

$$ \boxed{ \text{Episode }k\neq\text{Episode }k+1 } $$

for two reasons:

Reason 1 — policy changed
$$ \pi^k\rightarrow\pi^{k+1} $$
Reason 2 — stochastic environment
$$ D^k(t)\neq D^{k+1}(t) $$

in general.


One subtle but VERY important point about "history"

You might think:

"If the GRU uses history, does Episode \(k+1\) remember Episode \(k\)?"

No—not as an observation history.

The paper defines the policy history as:

$$ \bar{o}_t^i=(o_0^i,\ldots,o_t^i). $$

That means the GRU is learning temporal dependencies within the current decision trajectory.

So conceptually:

Episode k:

o₀ → o₁ → o₂ → ... → o₂₀₀
 │     │     │
 └──── GRU hidden state ────►


Episode k+1:

o₀ → o₁ → o₂ → ... → o₂₀₀
 │     │     │
 └──── GRU hidden state ────►

The previous episode's trajectory isn't appended to the next episode's observation history.

The paper does not explicitly state the hidden-state reset implementation, though; so the precise treatment of \(e_0\) at episode boundaries should be verified from the code if you're reproducing it.

What happens to the critic between the two episodes?

The critic is also updated.

So:

Episode \(k\)
$$ V^{i,k} $$

evaluates the trajectories.

Then HAPPO updates it:

$$ V^{i,k} \rightarrow V^{i,k+1}. $$
Episode \(k+1\)

The new critic:

$$ V^{i,k+1} $$

is used.

And because it is centralized during training:

V
i,k+1
=V
i,k+1
(
o
ˉ
1
,
o
ˉ
2
,
o
ˉ
3
,
o
ˉ
4
).

---

to reproduce the exact initial vector numerically for each actor at t=0 of every episode, e.g.

$$ o_0^1=[?, ?, ?, ?, ?, ?, ?], $$

then the paper alone is insufficient to give those exact numbers for the main \(M=4,L=4\) Merton experiment.

We know the structure of the vector, but the paper does not explicitly document the boundary initialization of the pre-\(t=0\) orders/shipments or the exact initial state for the typical experiment. The zero-initialization stated in Proposition 4 belongs to the simplified optimality test, not automatically to your main experiment.