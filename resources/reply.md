Yes. I checked this more deeply this time, including **the paper/supplement and the authors' official code repository**. This changes one important part of our earlier understanding: **the official implementation does use 5 parallel rollout environments.**

[Official HAPPO inventory-management code repository](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management?utm_source=chatgpt.com)

Below I answer your five questions specifically for the **\(M=4\), serial, \(L=4\), Merton, no-fixed-cost HAPPO benchmark behind Figure 9**.

---

# 1. Training steps and the "327 ± 35 episodes"

## 1.1 What is an episode?

The paper defines:

$$
\boxed{T=200}
$$

periods per episode. Appendix G explicitly gives episode length \(T=200\). 

The HAPPO training algorithm in the supplementary material is structured as:

1. initialize actor/critic networks;
2. get initial observations;
3. for \(t=0,\ldots,T-1\):

   * actors select actions;
   * environment executes them;
   * collect rewards and next observations;
4. use the collected trajectory to update the networks;
5. start the next episode/rollout. 

So conceptually:

$$
\boxed{
1\text{ episode}=200\text{ environment periods}
}
$$

---

## 1.2 Was it single-threaded?

**No — the official implementation uses 5 parallel rollout environments.**

The authors' `config.py` explicitly contains:

```text
n_rollout_threads = 5
```

with the description:

> "Number of parallel envs for training rollouts"

while:

```text
n_training_threads = 1
```

means **one PyTorch training thread**, not one environment. ([GitHub][1])

This distinction is extremely important:

### Environment simulation

$$
\boxed{5\text{ parallel environments}}
$$

### Neural-network training threads

$$
\boxed{1\text{ training thread}}
$$

So when we previously asked:

> "Was training done on 5 parallel threads?"

the more precise answer is:

> **Five parallel environment rollout workers were used, while the training computation itself was configured with one PyTorch training thread.**

The official repository's README output also explicitly shows:

```text
n_rollout_threads=5
n_training_threads=1
```

and even prints separate rewards for `thread 1` through `thread 5`. ([GitHub][2])

---

# 2. So what does 327 ± 35 actually mean with 5 environments?

This is where we need to be careful.

The **paper's Table 3** says:

$$
\boxed{327\pm35}
$$

for \(M=4\), Full Observation HAPPO.

It says these are the mean and 95% CI over **10 repeated training processes**. 

However, the paper itself does **not explicitly explain whether the reported "episodes" are counted as individual environment trajectories or rollout/update iterations when parallel environments are used**.

The official code makes the distinction visible.

The default configuration is:

$$
n_{\text{rollout threads}}=5
$$

$$
T=200
$$

$$
N_{\text{env steps}}=3,000,000.
$$

([GitHub][1])

So one rollout/update collects:

$$
5\times200=1000
$$

environment transitions.

And:

$$
\frac{3,000,000}{5\times200}
=
\boxed{3000}
$$

rollout/update iterations.

If you count **individual episode trajectories across all five environments**, that corresponds to:

$$
3000\times5
=
\boxed{15,000\text{ episode trajectories}}.
$$

This is why I would **not equate "327 ± 35 episodes" directly with the entire 3-million-step training budget** without qualification.

### The safest interpretation

* **Paper:** reports \(327\pm35\) "training episodes" as the convergence statistic.
* **Official implementation:** default training budget is **3,000,000 environment steps**, collected using **5 parallel environments**, with 200 steps per rollout.
* Therefore the full default run corresponds to **3000 rollout iterations / 15,000 individual environment episodes**.

The paper doesn't explicitly reconcile the terminology between Table 3 and the implementation's parallel-rollout counting.

---

# 3. What training budget produced Figure 9?

Here we can now give a much stronger answer.

The official code configuration specifies:

$$
\boxed{3,000,000\text{ environment steps}}
$$

as the default training budget. ([GitHub][1])

With:

$$
T=200
$$

and:

$$
5\text{ rollout environments},
$$

one update consumes:

$$
5\times200=1000
$$

environment steps.

Thus:

$$
\frac{3,000,000}{1000}
=
\boxed{3000\text{ rollout/update iterations}}.
$$

And across all five parallel environments:

$$
3000\times5
=
\boxed{15,000\text{ individual episode trajectories}}.
$$

### But one caveat

The paper does **not explicitly say in the Figure 9 paragraph**:

> "Figure 9 was generated after exactly 3,000,000 environment steps."

The **official code's default** is 3 million environment steps, and the code is explicitly provided by the authors as the implementation of the paper. The paper itself only gives the episode length and convergence statistics. ([GitHub][3])

So for a rigorous paper-reproduction statement I'd write:

> **The authors' released implementation uses a default training budget of 3 million environment steps, with five parallel rollout environments and 200 steps per episode. The paper itself reports \(327\pm35\) training episodes for the \(M=4\) Full-Observation condition but does not explicitly map that statistic to the 3-million-step budget.**

That's the defensible wording.

---

# 4. Initial state at \(t=0\)

Here the official code gives us the exact answer.

The serial environment defines:

$$
S_I=10
$$

$$
S_O=10
$$

$$
L=4.
$$

It then resets:

$$
I_0^i=10
$$

and:

$$
B_0^i=0
$$

and initializes the four pipeline positions as:

$$
[10,10,10,10].
$$

The code explicitly implements this reset. ([GitHub][4])

Therefore, for **every one of the four echelons**:

$$
\boxed{
I_0^i=10
}
$$

$$
\boxed{
B_0^i=0
}
$$

$$
\boxed{
[P_{-4}^i,P_{-3}^i,P_{-2}^i,P_{-1}^i]
=
[10,10,10,10]
}
$$

The reset observation is constructed as:

$$
[I_0^i,0,S_O,P_{-4}^i,P_{-3}^i,P_{-2}^i,P_{-1}^i].
$$

Since \(S_O=10\):

$$
\boxed{
o_0^i=[10,0,10,10,10,10,10]
}
$$

before normalization. ([GitHub][4])

Because:

$$
ACTION\_DIM=21,
$$

the code divides by:

$$
20.
$$

So the neural network receives:

$$
\boxed{
o_0^i=
[0.5,0,0.5,0.5,0.5,0.5,0.5].
}
$$

---

## One correction to our earlier interpretation

The code does **not** literally have a variable named \(O_{-1}\) stored separately.

At reset, `get_reset_obs()` inserts:

```python
S_O
```

as the third observation component.

So mathematically we can interpret it as the initialized pre-\(t=0\) downstream-order signal:

$$
O_{-1}=10,
$$

but implementation-wise it is simply the constant `S_O`.

Likewise, the pipeline is initialized directly as four 10s.

---

# 5. Figure 9 "overall cost" — how is ~34 calculated?

This is one of the most important findings from checking the actual code.

The **evaluation environment does not use the α-shaped reward for its reported cost**.

In training:

```python
processed_rewards =
ALPHA*i + (1-ALPHA)*mean(reward)
```

which corresponds to the shaped reward. ([GitHub][4])

But during evaluation:

```python
processed_rewards = [[i] for i in reward]
```

so the evaluation uses the **actual individual cost/reward**, not the α-shaped training reward. ([GitHub][4])

The raw per-period cost is:

$$
C_t^i
=
h_i I_t^i+b_iB_t^i
$$

plus ordering/fixed costs when those experiments activate them.

For the **no-fixed-cost** benchmark:

$$
h_i=b_i=1
$$

and:

$$
c_i=f_i=0
$$

so:

$$
\boxed{
C_t^i=I_t^i+B_t^i.
}
$$

The code computes the per-period reward as:

$$
-\;I_t^iH_i
-\;B_t^iB_i
-\text{ordering cost}
-\text{fixed cost}.
$$

With no fixed/ordering cost:

$$
r_t^i=-C_t^i.
$$

([GitHub][4])

### Does Figure 9 use \(\gamma=0.95\)?

**No, not for the reported evaluation cost.**

The \(\gamma=0.95\) value is an **RL training discount factor**. Appendix G lists it as a HAPPO hyperparameter. 

The evaluation environment uses raw per-period costs.

So the Figure 9 "Cost" is **not**:

$$
\sum_t\gamma^tC_t.
$$

---

# 6. Is it averaged over the 200 periods?

The implementation's evaluation is built around 200-period demand sequences/episodes, and the environment's evaluation reward is the raw period cost.

The authors' evaluation setup uses multiple demand traces, and the evaluation code reports average performance over them. The environment itself loads evaluation demand sequences, with each sequence being one evaluation episode. ([GitHub][4])

So conceptually the Figure 9 number is an **average undiscounted system cost per period**, aggregated over the evaluation trajectories:

$$
\boxed{
\text{Overall Cost}
\approx
\frac{1}{N_{\rm eval}}
\sum_{j=1}^{N_{\rm eval}}
\frac{1}{T}
\sum_{t=1}^{T}
\sum_{i=1}^{M}
C_{t,j}^i
}
$$

for the no-fixed-cost case.

For your case:

$$
M=4,\qquad T=200.
$$

Thus:

$$
\boxed{
\text{Cost}
\approx
\frac{1}{N_{\rm eval}}
\sum_j
\frac1{200}
\sum_{t=1}^{200}
\left(
I_{t,j}^1+B_{t,j}^1+
I_{t,j}^2+B_{t,j}^2+
I_{t,j}^3+B_{t,j}^3+
I_{t,j}^4+B_{t,j}^4
\right)
}
$$

### It is NOT divided by 4.

The paper defines:

$$
C_t^{total}=\sum_{i=1}^M C_t^i.
$$

So the reported system cost is the **sum across echelons**, not the average per echelon.

---

# 7. Are the first four periods discarded?

I found **no statement in the paper/supplement that says the first \(L=4\) periods are discarded as a warm-up**.

In fact, the official environment evaluates the episode as a 200-period sequence and does not show a special "discard first 4 periods" rule in the evaluation cost calculation.

So:

$$
\boxed{\text{Do not claim that the first 4 periods are discarded.}}
$$

They are part of the episode/evaluation trajectory unless a separate evaluation script adds a warm-up that is not documented in the paper.

This is important because your proposed formula:

$$
\frac1{200}\sum_{t=1}^{200}\sum_iC_t^i
$$

is much closer to the implementation than:

$$
\frac1{196}\sum_{t=5}^{200}\cdots.
$$

---

# 8. Alpha = 0.375 — which term gets the weight?

This one is completely clear.

The paper's Equation 7 is:

$$
\boxed{
r_t^i
=
-\alpha C_t^i
-
(1-\alpha)C_t^{total}
}
$$



Therefore:

$$
\boxed{\alpha=\text{weight on individual echelon cost}}
$$

and:

$$
\boxed{1-\alpha=\text{weight on total system cost}}.
$$

For:

$$
\alpha=0.375,
$$

we have:

$$
\boxed{
r_t^i
=
-0.375C_t^i
-
0.625C_t^{total}
}
$$

So your interpretation should be:

> **37.5% individual/self cost + 62.5% system-wide cost.**

This is also consistent with the implementation's reward processing:

```python
ALPHA*i + (1-ALPHA)*np.mean(reward)
```

because the environment's `reward` values are negative costs. ([GitHub][4])

---

# 9. Is there an algebraic formula for optimal α as a function of M?

**No.**

This is another point where we should be very careful.

The paper does **not derive** something like:

$$
\alpha^*=f(M).
$$

Instead, it performs controlled experiments over α and selects the value giving the lowest observed overall cost.

The resulting empirical values are:

$$
\boxed{
\begin{array}{c|c}
M&\alpha^*\\
\hline
3&0.5\\
4&0.375\\
5&0.75
\end{array}
}
$$



Notice something important:

$$
0.5\rightarrow0.375\rightarrow0.75
$$

is **not a monotonic mathematical function of \(M\)**.

So there is no legitimate formula such as:

$$
\alpha=\frac{1}{M}
$$

or:

$$
\alpha=\frac{M-1}{M}
$$

in the paper.

The correct statement is:

> **α is an empirically tuned hyperparameter selected through controlled experiments; the paper does not provide a closed-form optimal α as a function of M.**

---

# 10. PPO epochs — yes, it was 15

Here the official code gives us an answer that the paper's Appendix G table does not spell out.

The official `config.py` contains:

$$
\boxed{\texttt{ppo\_epoch}=15}
$$

with the description:

> "number of ppo epochs". ([GitHub][1])

It also uses:

$$
\boxed{\texttt{num\_mini\_batch}=1}
$$

so there is one minibatch per PPO epoch. ([GitHub][1])

Thus, per policy-update iteration, the PPO optimization is configured for:

$$
\boxed{15\text{ PPO epochs}}
$$

over the collected rollout data.

---

# 11. Was there learning-rate decay?

**No, not by default.**

The official configuration says:

```text
use_linear_lr_decay = False
```

and the configuration description says it controls whether a linear learning-rate schedule is used. ([GitHub][1])

So the default is:

$$
\boxed{\text{No linear learning-rate decay}}
$$

with:

$$
\boxed{lr=10^{-4}}
$$

for the actor and:

$$
\boxed{critic\_lr=10^{-4}}.
$$

([GitHub][1])

There is also no paper/supplement statement indicating another learning-rate schedule for the Figure 9 benchmark.

---

# 12. Was GRU data chunking used?

**Yes.**

This is another detail that is present in the official implementation but not in the paper's high-level Appendix G table.

The configuration specifies:

$$
\boxed{\texttt{data\_chunk\_length}=10}
$$

with the description:

> "Time length of chunks used to train a recurrent_policy." ([GitHub][1])

So the 200-step rollout is not simply treated as one giant 200-step recurrent sequence during optimization.

It is trained using recurrent chunks of:

$$
\boxed{10\text{ timesteps}}
$$

according to the code configuration.

Conceptually:

```text
200-step episode

t=0 ───────────────────────────────────────── t=199
       │        │        │        │
       ▼        ▼        ▼        ▼
     chunk    chunk    chunk    ... 
      10       10       10
```

giving approximately:

$$
\frac{200}{10}=20
$$

time chunks per environment trajectory.

---

# 13. One more important detail: recurrent configuration

The official code also has:

$$
\boxed{\texttt{recurrent\_N}=2}
$$

meaning two recurrent layers, matching the paper's:

$$
\boxed{\text{GRU layers}=2}.
$$

The hidden size is:

$$
\boxed{128}.
$$

([GitHub][1])

So the implementation aligns nicely with Appendix G:

| Parameter                |                     Paper / Appendix | Official code |
| ------------------------ | -----------------------------------: | ------------: |
| Episode length           |                                  200 |           200 |
| GRU layers               |                                    2 |             2 |
| GRU width                |                                  128 |           128 |
| \(\gamma\)               |                                 0.95 |          0.95 |
| Learning rate            |                          \(10^{-4}\) |   \(10^{-4}\) |
| PPO clip                 |                                  0.2 |           0.2 |
| Entropy coefficient      |                                 0.01 |          0.01 |
| PPO epochs               |                 **not in Table G.1** |        **15** |
| Minibatches              |              **not stated in paper** |         **1** |
| GRU chunk length         |              **not stated in paper** |        **10** |
| LR decay                 |                       **not stated** |     **False** |
| Training rollout envs    |              **not stated in paper** |         **5** |
| Training PyTorch threads |                       **not stated** |         **1** |
| Total env steps          | **not stated in paper's Appendix G** | **3,000,000** |

 ([GitHub][1])

---

# 14. So what is the *actual* reproduction configuration?

For your target:

$$
\boxed{
M=4,\quad L=4,\quad
\text{Merton},\quad
\text{no fixed cost},\quad
\alpha=0.375
}
$$

the implementation-level setup is:

```text
Environment
────────────────────────────────
Echelons                 4
Lead time                4
Episode length           200
Initial inventory        10 each
Initial backlog          0 each
Initial pipeline         [10,10,10,10]
Initial downstream order 10
Action dimension         21
Observation dimension    7
Merton demand            Yes
Fixed cost               No
Shipping loss            No


HAPPO
────────────────────────────────
Parallel rollout envs    5
Training threads         1
Total env steps          3,000,000
Rollout length           200
PPO epochs/update        15
Minibatches/update       1

GRU layers               2
GRU hidden size          128
Data chunk length        10

Actor LR                 1e-4
Critic LR                1e-4
Gamma                    0.95
GAE lambda               0.95
PPO clip                 0.2
Entropy coefficient      0.01
Linear LR decay          False

Alpha                    0.375
```

The official repository confirms the parallel-rollout and optimization settings. ([GitHub][2])

---

# 15. The biggest clarification from this investigation

There are actually **three different quantities that we should not mix up**:

### A. Paper's convergence statistic

$$
\boxed{327\pm35\text{ episodes}}
$$

10 repeated training processes. 

### B. Official implementation's rollout configuration

$$
\boxed{5\text{ parallel environments}}
$$

with:

$$
\boxed{200\text{ steps each}}
$$

so each rollout/update collects:

$$
1000\text{ environment steps}.
$$

### C. Official default full training budget

$$
\boxed{3,000,000\text{ environment steps}}
$$

which corresponds to:

$$
\boxed{3000\text{ rollout/update iterations}}
$$

or:

$$
\boxed{15,000\text{ individual episode trajectories across 5 workers}}.
$$

The paper doesn't explicitly explain how its **327 ± 35 episode convergence statistic** maps onto these parallel rollout units. Therefore, I would **not write in your thesis that "the paper trained for 327 episodes = 65,400 timesteps" anymore**. That was too simplistic once we checked the actual official implementation.

---

## Bottom line for your five questions

| Question                 | Answer                                                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Parallel workers?** | **Yes: 5 parallel rollout environments.** One PyTorch training thread.                                                                                                                                 |
| **Training budget?**     | Official code default: **3M environment steps**, 5 envs × 200 steps/rollout = 3000 rollout iterations = 15,000 individual trajectories.                                                                |
| **327 ± 35?**            | Paper's reported convergence statistic over 10 repeated training processes; its exact relationship to parallel rollout counting is **not explicitly reconciled**.                                      |
| **2. Initial state?**    | \(I_0=10,\ B_0=0,\) pipeline \(=[10,10,10,10]\), pre-\(t=0\) downstream-order field \(=10\). Normalized observation = `[0.5,0,0.5,0.5,0.5,0.5,0.5]`.                                                   |
| **3. Figure 9 cost?**    | **Undiscounted evaluation cost**, sum across 4 echelons, averaged over evaluation trajectories/periods; **not divided by 4**, and \(\gamma=.95\) is not used to discount the reported evaluation cost. |
| **Warm-up discarded?**   | **No documented \(L=4\) warm-up discard.**                                                                                                                                                             |
| **4. α=.375?**           | Weight on **individual cost**: \(r=-.375C_i-.625C^{total}\).                                                                                                                                           |
| **α=f(M)?**              | **No closed-form formula.** Empirically selected: .5, .375, .75 for M=3,4,5.                                                                                                                           |
| **5. PPO epochs?**       | **15**.                                                                                                                                                                                                |
| **LR decay?**            | **No**, `use_linear_lr_decay=False`.                                                                                                                                                                   |
| **GRU chunking?**        | **Yes, chunk length = 10** in official code.                                                                                                                                                           |

The official repository is especially valuable here because the paper itself says it provides extensible implementation code for the serial and network environments. ([Sage Journals][5])

One caveat worth keeping in your reproduction notes: the repository's **current `serial.py` defaults to `LEVEL_NUM=3` and `ALPHA=0.5`**, so for the \(M=4,\alpha=0.375\) Figure-9 condition those parameters must be changed; the repository's defaults are not themselves the \(M=4\) benchmark configuration. ([GitHub][4])

[1]: https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/config.py "Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/config.py at main · xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management · GitHub"
[2]: https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management "GitHub - xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management: Official codes for \"Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management: Reducing Costs and Alleviating Bullwhip Effect\" · GitHub"
[3]: https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management?utm_source=chatgpt.com "GitHub - xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management: Official codes for \"Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management: Reducing Costs and Alleviating Bullwhip Effect\" · GitHub"
[4]: https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/serial.py "Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/envs/serial.py at main · xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management · GitHub"
[5]: https://journals.sagepub.com/doi/full/10.1177/10591478241305863?utm_source=chatgpt.com "Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management - Xiaotian Liu, Ming Hu, Yijie Peng, Yaodong Yang, 2025"
