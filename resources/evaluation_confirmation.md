- Start with the cost of one echelon

For the no-fixed-cost experiment, the paper defines:

$$ C_t^i=h_iI_t^i+b_iB_t^i. $$

Here:

\(I_t^i\) = on-hand inventory at echelon \(i\)
\(B_t^i\) = backlog at echelon \(i\)
\(h_i\) = holding-cost coefficient
\(b_i\) = backlog-cost coefficient.

For the typical experiment:

$$ h_i=1,\qquad b_i=1. $$

Therefore:

$$ \boxed{ C_t^i=I_t^i+B_t^i } $$

for every echelon.

- Then add the four echelons

The total cost at period \(t\) is:

$$ \boxed{ C_t^{total} = \sum_{i=1}^{M}C_t^i } $$

For \(M=4\):

$$ \boxed{ C_t^{total} = C_t^1+C_t^2+C_t^3+C_t^4 } $$

or explicitly:

$$ C_t^{total} = (I_t^1+B_t^1) +(I_t^2+B_t^2) +(I_t^3+B_t^3) +(I_t^4+B_t^4). $$

So if, at some period, the four echelons had:

Echelon 1: I=4, B=1 → C¹=5
Echelon 2: I=6, B=0 → C²=6
Echelon 3: I=8, B=0 → C³=8
Echelon 4: I=10,B=0 → C⁴=10

then:

$$ C_t^{total}=5+6+8+10=29. $$
- Then accumulate this over the evaluation horizon

The important thing is that the economic evaluation is not the discounted RL objective.

The paper's theoretical RL objective uses:

$$ E\left[\sum_t\gamma^tC_t^{total}\right], $$

with:

$$ \gamma=0.95. $$

But the policy-comparison section says they use overall costs as their numerical comparison metric, and the demand-fluctuation section explicitly evaluates finite simulated periods.

The evaluation cost should therefore be understood as an undiscounted average cost over the simulated periods, rather than:

t
∑
	​

0.95
t
C
t
total
	​

.

---

It is NOT "34 per echelon"

This is a very common misunderstanding.

The value:

$$ 34 $$

is not:

$$ \frac{C^1+C^2+C^3+C^4}{4}. $$

The paper defines total cost as:

$$ C_t^{total}=\sum_{i=1}^{4}C_t^i. $$

Therefore the four echelons are added, not averaged.

So if the average costs were approximately:

$$ C^1=5,\quad C^2=7,\quad C^3=9,\quad C^4=13, $$

then:

$$ 5+7+9+13=34. $$

That's the kind of number represented by Figure 9.

---

Are the first four periods discarded?

The paper does not state that the first \(L=4\) periods are discarded as a warm-up.

The demand-fluctuation calculation explicitly uses the finite simulated periods:

$$ \{0,\ldots,T\}. $$

So there is no evidence in the paper that they calculate:

$$ t=5,\ldots,200 $$

and throw away the first four periods.

Therefore, for a faithful reproduction, you should not introduce a four-period warm-up discard unless the implementation you're reproducing explicitly does so.

---

Now let's move to Figure 10 — this is completely different

Figure 10 is measuring the bullwhip effect.

The paper defines the bullwhip effect as:

demand fluctuations increasing from downstream to upstream.

They use the coefficient of variation (CV) as the measure of demand fluctuation, following Fransoo and Wouters (2000).

So we're no longer measuring:

$$ I+B. $$

We're measuring:

$$ \boxed{ \text{variability of orders/demand} } $$
 What is "demand" for each echelon?

This is important.

For echelon \(i\), the demand it faces comes from the downstream actor.

The paper's formulation has:

$$ O_t^{i-1} $$

as the order received by echelon \(i\).

Therefore the demand sequence used to measure fluctuation at echelon \(i\) is:

$$ \boxed{ O_0^{i-1},O_1^{i-1},\ldots,O_T^{i-1} } $$

For \(M=4\):

Customer demand
       ↓
   Echelon 1
       ↓
   Echelon 2
       ↓
   Echelon 3
       ↓
   Echelon 4

The fluctuation is therefore measured on the ordering/demand stream passed from downstream to each echelon.

 Coefficient of variation

For echelon \(i\), define the average demand/order:

$$ \bar O^{i-1} = \frac{1}{T+1} \sum_{t=0}^{T}O_t^{i-1}. $$

Then the demand fluctuation measure \(F_i\) is the coefficient of variation:

$$ \boxed{ F_i = \frac{ \sqrt{ \frac{1}{T+1} \sum_{t=0}^{T} \left(O_t^{i-1}-\bar O^{i-1}\right)^2 } }{ \bar O^{i-1} } } $$

This is simply:

$$ \boxed{ F_i=\frac{\text{standard deviation of demand}}{\text{mean demand}} } $$

The paper explicitly says it measures demand fluctuation using the coefficient of variation and gives the finite-period formula in Section 3.3.

 Why divide by the mean?

Suppose:

Supply chain A
$$ \text{mean}=10,\qquad SD=5. $$

Then:

$$ CV=\frac5{10}=0.5. $$
Supply chain B
$$ \text{mean}=100,\qquad SD=50. $$

Then:

$$ CV=\frac{50}{100}=0.5. $$

Although B has a much larger absolute standard deviation, both have the same relative fluctuation.

That's why CV is useful for comparing demand variability across echelons.

 Now calculate \(F_i\) for every echelon

For your four-echelon system:

$$ F_1,\quad F_2,\quad F_3,\quad F_4. $$

Conceptually:

Customer → Echelon 1
              │
              │ F₁
              ▼
          Echelon 2
              │
              │ F₂
              ▼
          Echelon 3
              │
              │ F₃
              ▼
          Echelon 4
              │
              │ F₄
              ▼
        Manufacturer

As you move upstream, the ordering signal can become more variable.

That is the bullwhip effect.

 What is the "Average" value 0.758?

This is not the CV of only Echelon 1.

The paper calculates the average demand fluctuation across all \(M\) echelons as:

$$ \boxed{ F_{\text{Average}} = \frac1M \sum_{i=1}^{M}F_i } $$

This formula is explicitly given in the note under Table 2.

For \(M=4\):

$$ \boxed{ F_{\text{Average}} = \frac{F_1+F_2+F_3+F_4}{4} } $$

For HAPPO, the reported value is:

$$ \boxed{0.758}. $$

---

most important number: 92.37%

This is the bullwhip amplification.

The paper defines it as:

$$ \boxed{ \text{Amplification} = \frac{F_M}{F_1}-1 } $$

and reports it as a percentage.

For your \(M=4\):

$$ \boxed{ \text{Amplification} = \left( \frac{F_4}{F_1}-1 \right)\times100\%. } $$
 What does 92.37% actually mean?

It means:

$$ \frac{F_4}{F_1}-1=0.9237. $$

Therefore:

$$ \frac{F_4}{F_1}=1.9237. $$

So:

$$ \boxed{ F_4\approx1.924F_1. } $$

In plain English:

The relative demand fluctuation at the most upstream actor is about 1.924 times the fluctuation at the most downstream echelon.

Or:

The fluctuation has increased by about 92.37% from Echelon 1 to Echelon 4.

That is what the paper means by the 92.37% bullwhip amplification.

 Very important: 92.37% does NOT mean demand increased by 92%

This is another common misunderstanding.

It does not mean:

$$ Demand_4=1.9237\times Demand_1. $$

It means:

$$ \boxed{ CV_4=1.9237\times CV_1 } $$

approximately.

In other words, it is the variability that has amplified, not necessarily the average quantity demanded.

 Example of 92.37%

Suppose:

$$ F_1=0.50. $$

Then for 92.37% amplification:

$$ F_4 = 0.50(1+0.9237) $$ $$ =0.96185. $$

Then:

$$ \frac{0.96185}{0.50}-1 = 0.9237. $$

Therefore:

$$ \boxed{92.37\%}. $$

Again, this is just an illustrative calculation showing how the percentage works.

 Why use \(F_1\) and \(F_M\)?

Because they want to measure the end-to-end amplification.

For \(M=4\):

$$ Echelon\ 1 \rightarrow Echelon\ 2 \rightarrow Echelon\ 3 \rightarrow Echelon\ 4. $$

So:

$$ F_1 $$

is the fluctuation at the downstream end and:

$$ F_4 $$

is the fluctuation at the upstream end.

Therefore:

$$ \boxed{ \frac{F_4}{F_1}-1 } $$

measures how much the fluctuation has grown while traveling through the supply chain.