# 1. Validate and check if each of this is implemented in the project

The exact experiment we are trying to reproduce

The target is:

$$ \boxed{ M=4,\quad \text{Serial Supply Chain},\quad L=4 } $$

with the typical/no-fixed-cost configuration.

The core setup is:

Component	Base-paper setting
Supply-chain structure	Serial
Intermediate echelons	4
Lead time	4 periods
Demand	Merton Jump Diffusion
Holding cost \(h_i\)	1
Backlog cost \(b_i\)	1
Ordering cost \(c_i\)	Not applicable
Fixed cost \(f_i\)	Not applicable
Shipping loss \(\delta_i\)	Not applicable
Algorithm	HAPPO
Training structure	CTDE / Full Observation during training
Actor input	Local observation history
Critic input	All actors' observation histories
Network	GRU-based RNN
GRU layers	2
GRU width	128
Episode length	200 periods
Discount \(\gamma\)	0.95
Learning rate	\(10^{-4}\)
PPO clip \(\epsilon\)	0.2
Entropy coefficient \(\beta\)	0.01
\(\alpha\)	0.375 for \(M=4\)
Training episodes reported	327 ± 35 for Full-Observation HAPPO

The GRU and HAPPO hyperparameters are explicitly given in Table G.1, and the Merton parameters are given in Table G.2.

---

# 2. Validate and check if each of this is implemented in the project

GRU configuration

The paper gives:

$$ \boxed{\text{GRU layers}=2} $$ $$ \boxed{\text{GRU width}=128} $$

So each actor/critic uses a 2-layer GRU with hidden width 128.

The remaining hyperparameters are:

Hyperparameter	Value
Episode length \(T\)	200
GRU layers	2
GRU width	128
\(\gamma\)	0.95
Learning rate	\(10^{-4}\)
PPO clip \(\epsilon\)	0.2
Entropy coefficient \(\beta\)	0.01

The paper explicitly says these hyperparameters are the same for actor and critic.

---

# 3. Validate and check if each of this is implemented in the project

Customer demand: Merton Jump Diffusion

This is the external disturbance entering the system.

The customer is actor 0, and its demand is:

$$ O_t^0=\left\lfloor \ell_1 e^{J_t}\right\rfloor. $$

The Merton process is:

$$ J_t = J_{t-1} + \left(\mu-\frac{\sigma^2}{2}\right) + \sigma Z_t + \ell_2N_t + \ell_3N_tZ'_t. $$

For the serial supply-chain experiment, the parameters are:

$$ \boxed{ \mu=5\times10^{-5} } $$ $$ \boxed{ \sigma=10^{-2} } $$ $$ \boxed{ \lambda=15 } $$ $$ \boxed{ \ell_1=10 } $$ $$ \boxed{ \ell_2=0 } $$ $$ \boxed{ \ell_3=0.01 } $$

where:

$$ N_t\sim Poisson(15) $$

and \(Z_t,Z'_t\) are standard normal random variables.

Therefore:

$$ \boxed{ O_t^0= \left\lfloor10e^{J_t}\right\rfloor } $$

is what ultimately enters Echelon 1 as customer demand.

---

# 4. Validate and check if each of this is implemented in the project

POMG

The whole supply chain is formulated as a:

$$ \boxed{\text{Partially Observable Markov Game}} $$

because we have four interacting decision-making agents.

Each echelon is an agent.

So:

$$ M=4 $$

means:

$$ \boxed{ Agent_1,Agent_2,Agent_3,Agent_4 } $$

What does each agent observe?

This is one of the most important parts.

Echelon \(i\)'s observation is:

$$ o_t^i= \left( I_t^i, B_t^i, O_{t-1}^{i-1}, P_{t-L}^i,\ldots,P_{t-1}^i \right). $$

For:

$$ L=4 $$

this becomes approximately:

$$ \boxed{ o_t^i= (I_t^i,B_t^i,O_{t-1}^{i-1}, P_{t-4}^i,P_{t-3}^i,P_{t-2}^i,P_{t-1}^i) } $$

So each actor sees:

its own current inventory;
its own backlog;
the previous-period order from its downstream actor;
its recent shipment/in-transit history.

---

# 5. Validate and check if each of this is implemented in the project

GRU doesn't just see one observation

This is where the architecture becomes important.

Instead of:

$$ a_t^i=\pi(o_t^i), $$

the paper uses:

$$ \boxed{ a_t^i \sim \pi^i(\cdot|\bar{o}_t^i) } $$

where:

$$ \bar{o}_t^i= (o_0^i,o_1^i,\ldots,o_t^i). $$

So the actor is effectively using historical observations.

Why?

Because inventory control has temporal dependencies.

For example:

Demand spike
      ↓
Echelon 1 orders more
      ↓
Echelon 2 sees larger order
      ↓
Echelon 3 reacts
      ↓
Echelon 4 reacts
      ↓
Inventory arrives 4 periods later

The current observation alone may not tell the whole story.

The GRU gives the network a memory of previous observations.

The paper explicitly motivates the GRU because simple fully connected networks using only current observations can neglect demand fluctuations and inventory changes over time.

The action

Each echelon's action is:

$$ \boxed{ a_t^i=O_t^i } $$

i.e. the ordering quantity.

The action space is:

$$ \boxed{ A^i=\{0,1,\ldots,O_{\max}\} } $$

So the agent isn't directly deciding:

"Ship 12 units."

It decides:

"I want to order 12 units from my upstream supplier."

The actual shipment can be constrained by what the upstream echelon has available.

---

# One crucial caveat for own implementation

There are several things the paper does not give enough detail to reproduce exactly.

For example, from the paper alone we should not invent:

exact fully-connected-layer dimensions;
exact PPO minibatch size;
exact number of PPO epochs per update;
exact optimizer implementation details beyond the reported learning rate;
exact initialization scheme;
exact action \(O_{\max}\) unless specified elsewhere in the implementation;
exact number of gradient updates corresponding to 327 episodes.

The paper gives the GRU structure and major hyperparameters, but not every low-level implementation detail.