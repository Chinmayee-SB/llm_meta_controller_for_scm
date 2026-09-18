# 1. Validate and check if each of this is implemented in the project

HAPPO architecture

This is the central architecture.

There are four actors.

Each actor has:

an actor/policy network;
a critic/value network.

So conceptually:

                   SUPPLY CHAIN
                       │
          ┌────────────┼────────────┐
          │            │            │
       Actor 1      Actor 2      Actor 3 ... Actor 4
          │            │            │
        GRU          GRU          GRU        GRU
          │            │            │
       Action       Action       Action     Action
          │            │            │
          └────────────┼────────────┘
                       │
                  Environment
                       │
                  Cost / Reward
                       │
                       ▼
               Centralized Critics

The paper uses HAPPO, which applies an on-policy PPO process to each actor and follows the CTDE framework.

---

# 2. Validate and check if each of this is implemented in the project

Actor network

For each actor:

$$ o_t^i,e_t^i \rightarrow GRU \rightarrow e_{t+1}^i,\text{features} $$

The GRU receives:

current observation \(o_t^i\)
previous hidden state \(e_t^i\)

and produces:

updated hidden state \(e_{t+1}^i\)
high-dimensional features.

Then:

$$ \text{GRU features} \rightarrow \text{Fully Connected layer} \rightarrow \text{Softmax} $$

The Softmax gives:

$$ P(a_t^i=0),P(a_t^i=1),\ldots,P(a_t^i=O_{\max}). $$

The agent then samples an ordering action from this distribution.

This actor architecture is explicitly described in Section 3.2.

---

# 3. Validate and check if each of this is implemented in the project

Critic network

The critic also uses the GRU architecture.

But there is an important difference.

Actor
GRU
 ↓
Features
 ↓
Fully Connected
 ↓
Softmax
 ↓
Action probabilities
Critic
GRU
 ↓
Features
 ↓
Value estimate

There is no Softmax at the end of the critic.

The critic outputs a scalar state/value estimate.

The paper explicitly states that this is the primary architectural difference between actor and critic.

---

# 4. Validate and check if each of this is implemented in the project

CTDE?

CTDE =

Centralized Training, Decentralized Execution

During training:

Actor

Actor \(i\):

$$ \boxed{ \text{local observation history} } $$
Critic

Critic \(i\):

$$ \boxed{ \text{observation histories of all actors} } $$

So for \(M=4\):

$$ V^i = V^i( \bar{o}^1, \bar{o}^2, \bar{o}^3, \bar{o}^4 ) $$

This gives the critic information about the whole supply chain during training.

But when actually operating the supply chain, each actor doesn't need the global state.

It acts using its own local observation history.

That's the decentralized execution part.

OUR target:

HAPPO — Full Observation during training

Actor 1 → local info
Actor 2 → local info
Actor 3 → local info
Actor 4 → local info

          ↓

Critics receive all observation histories

This is the standard CTDE setting.