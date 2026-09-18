# 1. Validate and check if each of this is implemented in the project

Important distinction: ORDER vs SHIPMENT

This is a common point of confusion.

Suppose:

$$ O_t^1=20. $$

That means Echelon 1 orders 20.

It does not necessarily mean Echelon 1 receives 20.

The actual shipment is:

$$ P_t^i = \min \left( B_t^i+O_t^{i-1}, I_t^i+P_{t-L}^i \right). $$

For \(L=4\):

$$ \boxed{ P_t^i= \min \left( B_t^i+O_t^{i-1}, I_t^i+P_{t-4}^i \right) } $$

So supply availability constrains what can actually be shipped.

The manufacturer is assumed to have infinite inventory, so the most upstream echelon's ordering is not constrained by manufacturer inventory.

---

# 2. Validate and check if each of this is implemented in the project

Inventory dynamics

Inventory evolves as:

$$ I_{t+1}^i = I_t^i + P_{t-L}^i - P_t^i. $$

With \(L=4\):

$$ \boxed{ I_{t+1}^i = I_t^i+ P_{t-4}^i-P_t^i } $$

So:

Beginning inventory
        +
Shipment arriving after 4 periods
        -
Shipment sent out
        =
Next-period inventory


Backlog dynamics

Backlog evolves according to:

$$ B_{t+1}^i = B_t^i + O_t^{i-1} - P_t^i. $$

So:

Previous backlog
      +
new downstream demand/order
      -
what you actually shipped
      =
new backlog

If an echelon receives an order for 20 but can ship only 15:

$$ B_{t+1}=B_t+20-15 $$

so backlog increases by 5.

---

# 3. Validate and check if each of this is implemented in the project

- Cost function

For the typical experiment:

$$ h_i=1 $$

and:

$$ b_i=1. $$

The paper defines:

$$ C_t^i=h_iI_t^i+b_iB_t^i. $$

Therefore:

$$ \boxed{ C_t^i=I_t^i+B_t^i } $$

for your setup.

So the agent pays for:

Holding inventory

Too much inventory:

$$ I_t^i\uparrow \Rightarrow C_t^i\uparrow $$
Backlog

Too little inventory:

$$ B_t^i\uparrow \Rightarrow C_t^i\uparrow $$

The goal is therefore to find the right balance.

- Total supply-chain cost

The system cost is:

$$ C_t^{total} = \sum_{i=1}^{M}C_t^i. $$

For \(M=4\):

$$ \boxed{ C_t^{total} = C_t^1+C_t^2+C_t^3+C_t^4 } $$

or:

$$ C_t^{total} = \sum_{i=1}^{4} (I_t^i+B_t^i). $$

This is ultimately what we care about from a supply-chain perspective.

- The reward

This is where \(\alpha\) enters.

Each actor receives:

$$ r_t^i = -\alpha C_t^i - (1-\alpha)C_t^{total}. $$

For your \(M=4\) setup:

$$ \boxed{\alpha=0.375} $$

so:

$$ \boxed{ r_t^i = -0.375C_t^i - 0.625C_t^{total} } $$

Therefore every actor considers:

37.5% own cost
62.5% overall system cost

in its reward.

The paper's controlled α experiment selected \(0.375\) for the four-echelon serial system.

---
