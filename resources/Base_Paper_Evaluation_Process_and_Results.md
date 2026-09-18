The base paper had this as their evaluation process and results

- Merton Demand + No Fixed Cost + M=4 bar - 34 is an approximate value read from the plotted bar
- With h = 1 and b = 1, the per-period system cost is:

$$ C_t^{total} = \sum_{i=1}^{4} (I_t^i+B_t^i) $$

So the 34 overall cost represents the aggregate holding + backlog cost incurred across the four echelons, according to the paper's evaluation procedure.

- For comparison, in the same M=4, MJD, no-fixed-cost setting, Figure 9 shows approximately:

HAPPO: ~34
PPO: ~37
Non-stationary Base Stock: ~42.4

---
# After training, they evaluate the policy

This is the critical distinction:

**Training cost ≠ final reported evaluation cost.**

After HAPPO has learned the policy, they evaluate the trained policy separately.

The paper evaluates policies on:

$$
\boxed{20\text{ different demand traces}}
$$

and repeats HAPPO/PPO training **10 times** to account for training randomness. 

So the evaluation looks like:

```text
             TRAINING
                 │
       ┌─────────┴─────────┐
       │                   │
    HAPPO run 1         HAPPO run 10
       │                   │
       ▼                   ▼
   trained policy       trained policy
       │                   │
       └─────────┬─────────┘
                 ↓
              TESTING
                 ↓
        20 demand traces
```

---

# Calculate the cost on each test trace

Suppose one test demand trace has 200 periods.

At each period:

$$
C_t^{total}
=
\sum_{i=1}^{4}(I_t^i+B_t^i)
$$

Then the evaluation aggregates the resulting costs over the simulation horizon.

Conceptually:

$$
C_{\text{trace}}
=
\text{aggregate of }C_1,C_2,\ldots,C_{200}
$$

The exact implementation/plotting convention for the Figure 9 cost bars is **not spelled out as a single algebraic equation in the paper**, so we should not invent a more precise normalization than the paper provides.

---

# Repeat across the 20 demand traces

Now imagine the evaluation produces something like:

```text
Trace 1  → cost
Trace 2  → cost
Trace 3  → cost
...
Trace 20 → cost
```

Then they average the evaluation results.

So the final number represents the policy's **average performance under the stochastic Merton demand environment**, rather than the cost from one particular demand sequence.

---

# Then Figure 9 gives approximately 34

For:

```text
Merton Demand
+
No Fixed Cost
+
M = 4
+
HAPPO
```

Figure 9 shows approximately:

$$
\boxed{\text{HAPPO overall cost}\approx34}
$$

The corresponding approximate comparison is:

| Method                    | Overall cost |
| ------------------------- | -----------: |
| **HAPPO**                 |      **≈34** |
| PPO                       |          ≈37 |
| Non-stationary Base Stock |        ≈42.4 |

The important conclusion is:

$$
34 < 37 < 42.4
$$

so HAPPO achieves the lowest overall cost among these methods for this particular benchmark. 

---

# Other metric was Average fluctuation 

 **0.758 is not a cost**. It is a measure of **how much the demand/order quantity fluctuates at the four echelons** under the trained HAPPO policy.

The paper defines the overall average fluctuation as:

$$
\boxed{F=\frac{1}{M}\sum_{i=1}^{M}F_i}
$$

For your case:

$$
M=4
$$

so:

$$
\boxed{F=\frac{F_1+F_2+F_3+F_4}{4}}
$$

The paper reports:

$$
\boxed{F_{\text{HAPPO}}=0.758}
$$

for **Merton demand + no fixed cost + 4 echelons**. 

---

# First: what does "fluctuation" mean?

Think about the customer demand entering the supply chain.

Suppose the customer demand is:

```text
10, 12, 11, 15, 13, 20, 18, ...
```

The retailer doesn't necessarily order exactly:

```text
10, 12, 11, 15, 13, 20, 18
```

It might order:

```text
11, 13, 12, 16, 15, 25, 20
```

Then the distributor might order:

```text
12, 14, 13, 18, 17, 29, 23
```

And upstream the fluctuations may become even larger.

That's the **bullwhip effect**:

```text
Customer demand
      │
      ▼
Echelon 1
   small fluctuation
      │
      ▼
Echelon 2
   larger fluctuation
      │
      ▼
Echelon 3
   larger fluctuation
      │
      ▼
Echelon 4
   potentially much larger fluctuation
```

The paper measures this behavior using **demand fluctuation \(F_i\)** for each echelon. 

---

# What exactly is \(F_i\)?

This is the important part.

For each echelon \(i\), the paper calculates a **demand fluctuation measurement \(F_i\)** based on the variability of the order/demand sequence at that echelon.

The paper then uses those \(F_i\)'s to calculate two metrics:

### Average fluctuation

$$
\boxed{
F=\frac{1}{M}\sum_{i=1}^{M}F_i
}
$$

### Amplification

$$
\boxed{
\text{Amplification}
=
\frac{F_M}{F_1}-1
}
$$

where:

* \(F_1\) = fluctuation at the downstream/first echelon
* \(F_M\) = fluctuation at the most upstream/final echelon.



**Important limitation from the paper:** the table reports the resulting average \(F\) and amplification, but it does **not give the four individual \(F_1,F_2,F_3,F_4\) values for the HAPPO 0.758 row**. So we should not invent those four values.

---

# Step 2 — Customer demand enters Echelon 1

The customer is effectively outside the four-echelon system.

So:

```text
Customer demand
       │
       ▼
┌────────────┐
│ Echelon 1  │
└────────────┘
```

Echelon 1 has to respond to that demand.

It observes its state and makes an ordering decision using its HAPPO policy.

---
# Step 5 — Run the trained policy over the whole simulation

This happens repeatedly:

```text
t = 1
   ↓
Observe
   ↓
Order
   ↓
Inventory / backlog update
   ↓

t = 2
   ↓
Observe
   ↓
Order
   ↓
Inventory / backlog update

...

t = 200
```

The paper's standard episode length is:

$$
T=200
$$

timesteps. 

During this process, the simulator records the order/demand behavior at each echelon.

---

# Step 6 — Now calculate fluctuation at each echelon

This is where \(F_i\) comes from.

After the simulation, you have sequences corresponding to each echelon:

```text
Echelon 1:
O₁(1), O₁(2), O₁(3), ..., O₁(T)

Echelon 2:
O₂(1), O₂(2), O₂(3), ..., O₂(T)

Echelon 3:
O₃(1), O₃(2), O₃(3), ..., O₃(T)

Echelon 4:
O₄(1), O₄(2), O₄(3), ..., O₄(T)
```

The paper's fluctuation metric is then applied to these sequences to obtain:

$$
F_1,F_2,F_3,F_4
$$

The critical idea is:

> **\(F_i\) tells us how much the demand/order stream fluctuates at echelon \(i\).**

---

# 10. Step 7 — Average the four echelon fluctuations

Once the four values are obtained:

$$
F_1,F_2,F_3,F_4
$$

the paper calculates:

$$
F=
\frac{F_1+F_2+F_3+F_4}{4}
$$

For example, **if** the four values happened to be:

$$
0.60,\;0.70,\;0.80,\;0.94
$$

then:

$$
F=
\frac{0.60+0.70+0.80+0.94}{4}
$$

$$
F=0.76
$$

That would produce approximately the same scale as the reported 0.758.

**But those four numbers are not the paper's actual values**—I'm using them only to demonstrate how the averaging works.

The paper does not provide the individual four HAPPO values needed to reconstruct exactly 0.758 from the published table.

---

# Step 8 — Repeat over evaluation demand traces

The paper evaluates the policies on:

$$
\boxed{20\text{ different demand traces}}
$$

So this isn't based on just one MJD demand realization.

Conceptually:

```text
MJD Trace 1
    ↓
F₁,F₂,F₃,F₄
    ↓
Average F

MJD Trace 2
    ↓
F₁,F₂,F₃,F₄
    ↓
Average F

...

MJD Trace 20
    ↓
F₁,F₂,F₃,F₄
    ↓
Average F
```

Then those evaluation results are aggregated to produce the reported metric.

HAPPO/PPO training itself is repeated 10 times to account for training randomness. 

---

# Finally → 0.758

After running the trained HAPPO policy under the Merton-demand evaluation setting, calculating the echelon-level fluctuations and averaging across the four echelons/evaluation traces, the paper reports:

$$
\boxed{
F_{\text{HAPPO}}=0.758
}
$$

for:

> **Merton Demands + No Fixed Cost + 4 echelons**

---
# Other metric was Bullwhip amplification

Yes. This is the **bullwhip part** of the evaluation, and the key is that **92.37% is not the same thing as the average fluctuation 0.758**.

For your exact benchmark, the paper reports:

$$
\boxed{\text{HAPPO Bullwhip Amplification}=92.37\%}
$$

under **Merton demand + 4 echelons + no fixed cost**. 

---

# 1. What is Bullwhip Amplification?

The **bullwhip effect** describes a phenomenon where:

> A relatively small fluctuation in customer demand becomes progressively larger as we move upstream through the supply chain.

Imagine:

```text
Customer
Demand
  │
  │  small fluctuation
  ▼
Echelon 1
  │
  │  larger fluctuation
  ▼
Echelon 2
  │
  │  even larger
  ▼
Echelon 3
  │
  │  much larger
  ▼
Echelon 4
```

So the supply chain can effectively **amplify demand variability**.

---

# 2. How does this paper measure it?

The paper first calculates a fluctuation metric \(F_i\) for each echelon.

So for your four-echelon system:

$$
F_1,\;F_2,\;F_3,\;F_4
$$

Then the paper defines **bullwhip amplification** as:

$$
\boxed{
\text{Amplification}
=
\frac{F_M}{F_1}-1
}
$$

where:

* \(F_1\) = fluctuation at the first/downstream echelon
* \(F_M\) = fluctuation at the last/upstream echelon
* \(M\) = number of echelons

For your case:

$$
M=4
$$

therefore:

$$
\boxed{
A=\frac{F_4}{F_1}-1
}
$$



---

# 3. What does 92.37% mean?

The paper reports:

$$
\boxed{A=92.37\%}
$$

This means that the fluctuation at the **most upstream echelon is approximately 92.37% greater than the fluctuation at the first echelon**, according to the paper's fluctuation metric.

In other words:

$$
F_4\approx1.9237F_1
$$

because:

$$
1+0.9237=1.9237
$$

So if, purely for illustration:

$$
F_1=1
$$

then:

$$
F_4\approx1.9237
$$

The upstream fluctuation would be about **1.92 times** the downstream fluctuation.

### Important:

Do **not** say:

> "92.37% of the demand is amplified."

Instead say:

> **"The upstream fluctuation is 92.37% higher than the downstream fluctuation."**

---

# 9. Step 5 — Run the supply-chain simulation

At every timestep:

```text
Observe state
     ↓
HAPPO chooses order
     ↓
Orders/shipment propagate
     ↓
Lead-time effects applied
     ↓
Inventory changes
     ↓
Backlog changes
     ↓
Next timestep
```

The paper uses episodes of:

$$
T=200
$$

timesteps. 

During this simulation, the order/demand behavior at each echelon is recorded.

---

# 10. Step 6 — Obtain the fluctuation \(F_i\)

Now we measure the fluctuation at each echelon.

You can think of it as:

```text
Echelon 1
Demand/order sequence
        ↓
    fluctuation
        ↓
       F₁

Echelon 2
Demand/order sequence
        ↓
    fluctuation
        ↓
       F₂

Echelon 3
Demand/order sequence
        ↓
    fluctuation
        ↓
       F₃

Echelon 4
Demand/order sequence
        ↓
    fluctuation
        ↓
       F₄
```

The important thing is that **\(F_i\) is a metric calculated from the demand/order variability at that echelon**.

---

# 11. Step 7 — Average fluctuation is a separate calculation

Earlier we discussed:

$$
F_{\text{average}}
=
\frac{F_1+F_2+F_3+F_4}{4}
$$

The paper reports:

$$
\boxed{F_{\text{average}}=0.758}
$$

for HAPPO in this configuration. 

But **92.37% does NOT come from 0.758 directly**.

This is extremely important.

You have two different calculations:

### Average fluctuation

$$
\boxed{
\frac{F_1+F_2+F_3+F_4}{4}
=0.758
}
$$

### Bullwhip amplification

$$
\boxed{
\frac{F_4}{F_1}-1
=0.9237
}
$$

then:

$$
0.9237\times100
=
92.37\%
$$

---

# 12. So where exactly does 92.37% come from?

The calculation is:

$$
\boxed{
\text{Bullwhip Amplification}
=
\left(
\frac{\text{upstream fluctuation}}
{\text{downstream fluctuation}}
-1
\right)\times100
}
$$

For four echelons:

$$
\boxed{
A=
\left(
\frac{F_4}{F_1}-1
\right)\times100
}
$$

The paper's result is:

$$
\boxed{A=92.37\%}
$$

Therefore:

$$
\frac{F_4}{F_1}-1=0.9237
$$

and:

$$
\frac{F_4}{F_1}=1.9237
$$

So:

$$
\boxed{F_4=1.9237F_1}
$$

---

# 13. Can we calculate \(F_1\) and \(F_4\) from the paper's 0.758 and 92.37%?

**No—not uniquely.**

This is an important caveat.

We know:

$$
\frac{F_1+F_2+F_3+F_4}{4}=0.758
$$

and:

$$
\frac{F_4}{F_1}-1=0.9237
$$

But the paper's published table does **not provide the individual values \(F_1,F_2,F_3,F_4\)** for this HAPPO row.

Therefore we cannot legitimately say:

> "HAPPO had \(F_1=X\) and \(F_4=Y\)."

unless we have the underlying experiment output/data.

There are infinitely many combinations of \(F_1,F_2,F_3,F_4\) that could satisfy those two reported metrics.

So the correct explanation is based on the **metric definition**, not an invented reconstruction.

---

# 14. Step 8 — Repeat over evaluation traces

The evaluation isn't based on one MJD demand realization.

The paper evaluates policies using:

$$
\boxed{20\text{ different demand traces}}
$$

and HAPPO/PPO training is repeated 10 times to account for training randomness. 

Conceptually:

```text
        HAPPO trained policy
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
     Trace 1   Trace 2   ...
        │        │
        ↓        ↓
   F₁,F₂,F₃,F₄  F₁,F₂,F₃,F₄
        │        │
        ↓        ↓
   amplification
        │
        └────────┬─────────
                 ↓
        Aggregate evaluation
                 ↓
             92.37%
```

The paper reports the resulting benchmark statistic in Table 2.

---

# 15. Compare HAPPO against PPO

This is where the result becomes meaningful.

The paper reports:

| Method                    | Average fluctuation | Bullwhip amplification |
| ------------------------- | ------------------: | ---------------------: |
| **HAPPO**                 |           **0.758** |             **92.37%** |
| PPO                       |               0.890 |                 122.8% |
| Non-Stationary Base Stock |               0.764 |                 101.9% |



Therefore:

$$
92.37\% < 122.8\%
$$

HAPPO has substantially less upstream amplification than PPO in this particular benchmark.

The reduction relative to PPO is:

$$
\frac{122.8-92.37}{122.8}\times100
$$

$$
\approx24.8\%
$$

So you can say:

> **HAPPO reduces the reported bullwhip amplification by approximately 24.8% relative to PPO under the same Merton/no-fixed-cost/4-echelon benchmark.**

---

