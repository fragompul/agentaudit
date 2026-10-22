# Mathematics

Notation: agents $m=1..M$, tasks $t=1..T$, runs $r=1..R$, scores $Y_{mtr}\in[0,1]$.
Tasks are treated as an i.i.d. sample from a task population; this is the assumption behind every interval and test.

## pass@k and pass^k

If task $t$ succeeds with probability $p$ in each independent attempt and we observed $c$ successes in $n$ attempts, then

$$\widehat{p^k}=\frac{\binom{c}{k}}{\binom{n}{k}},\qquad
\widehat{1-(1-p)^k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$

are unbiased: $\binom{c}{k}/\binom{n}{k}$ is the probability that $k$ attempts drawn without replacement from the $n$ all succeed, and
$E[\binom{C}{k}]=\binom{n}{k}p^k$ for $C\sim\mathrm{Bin}(n,p)$. pass^k (the reliability metric of tau-bench) is decreasing in $k$,
pass@k increasing, and both equal $p$ at $k=1$. The plug-in $\hat p^k$ is biased upward.

## Why pooled binomial intervals are wrong

Let $p_t$ be the success probability of task $t$ with variance $\tau^2$ over tasks and mean $p$. The benchmark mean has variance

$$\operatorname{Var}(\bar Y)=\frac{\tau^2}{T}+\frac{p(1-p)-\tau^2}{TR}.$$

The pooled binomial interval assumes $p(1-p)/(TR)$. The ratio is the *design effect*

$$\mathrm{DEFF}=1+(R-1)\rho,\qquad \rho=\frac{\tau^2}{p(1-p)}.$$

A nominal 95% pooled interval therefore has coverage $2\Phi(1.96/\sqrt{\mathrm{DEFF}})-1$, which falls as $R$ grows
(see the [simulation results](results.md)). The only valid fix is to treat the *task* as the sampling unit.

## Bootstrap intervals

`bootstrap_ci` supports three resampling schemes for the $(T\times R)$ matrix:

* **tasks** (cluster bootstrap): resample rows. Consistent for the mean; the default.
* **runs only**: resample within rows. Ignores $\tau^2/T$ and is anti-conservative.
* **two-stage**: resample rows, then runs within rows. Counts run noise twice
  ($\approx \tau^2/T + 2\sigma^2_e/(TR)$) and is conservative.

BCa uses a jackknife over tasks for the acceleration constant. The profile-likelihood beta-binomial interval models
$p_t\sim\mathrm{Beta}(\mu\kappa,(1-\mu)\kappa)$ and profiles out $\kappa$.

## Variance decomposition (generalizability theory)

$Y_{mtr}=\mu+a_m+b_t+(ab)_{mt}+e_{mtr}$ with variances $\sigma^2_m,\sigma^2_t,\sigma^2_{mt},\sigma^2_e$. For balanced data the
expected mean squares are

$$E[MS_m]=\sigma^2_e+R\sigma^2_{mt}+TR\sigma^2_m,\quad
E[MS_t]=\sigma^2_e+R\sigma^2_{mt}+MR\sigma^2_t,$$
$$E[MS_{mt}]=\sigma^2_e+R\sigma^2_{mt},\quad E[MS_e]=\sigma^2_e,$$

solved for the components (negative estimates are truncated at 0). The variance of the difference between two agents' means is

$$\operatorname{Var}(\bar Y_a-\bar Y_b)=2\left(\frac{\sigma^2_{mt}}{T}+\frac{\sigma^2_e}{TR}\right),$$

because task effects cancel in a paired comparison. The generalizability coefficient
$E\rho^2=\sigma^2_m/(\sigma^2_m+\sigma^2_{mt}/T+\sigma^2_e/(TR))$ is the share of observed between-agent variance that is true skill.
For binary scores this is a linear-probability decomposition: exact for variances of means, not latent-scale.

## Paired comparison

For agents A and B let $d_t=\bar Y_{At}-\bar Y_{Bt}$. The task is the unit: all tests operate on $d_1..d_T$.

* **Sign-flip permutation test**: under $H_0$ the distribution of each $d_t$ is symmetric about 0; exact enumeration for $\le 14$ nonzero differences, Monte Carlo otherwise.
* **Cluster bootstrap** of $\bar d$ gives the interval and a p-value.
* **Exact McNemar** on per-task majority outcomes: two-sided binomial test on discordant tasks. With $R=1$ it is the natural test.
* **Probability of superiority** $P(d>0)+\tfrac12 P(d=0)$ and Cohen's $d_z=\bar d/s_d$.

Pooling all $TR$ run pairs inflates the effective sample size and the type-I error when agent-by-task interaction is present.
**Multiplicity**: Holm (strong FWER) and Benjamini-Hochberg (FDR).

## Ranking uncertainty

Resample tasks (the same tasks for every agent), recompute scores and ranks. Ties are broken uniformly at random so each
replicate assigns every rank once. Outputs: rank distributions, $P(\text{top-}k)$, and the matrix of $P(\bar Y_i>\bar Y_j)$.

## Power and allocation

With cost $c_\text{task}$ per new task and $c_\text{run}$ per run, budget $B=T(c_\text{task}+Rc_\text{run})$,

$$\operatorname{Var}(\bar d)=\frac{s_t+s_e/R}{T},\qquad s_t=2\sigma^2_{mt},\ s_e=2\sigma^2_e.$$

At fixed *attempts* $TR$ the variance is $(R s_t+s_e)/(TR)$, increasing in $R$: **with free tasks, more tasks always beat more repeats**.
When tasks are costly the minimiser is $R^*=\sqrt{s_e c_\text{task}/(s_t c_\text{run})}$. Power uses the noncentral $t$ with $T-1$ degrees of freedom.
Repeats remain essential for *reliability* metrics (pass^k) and for estimating $\sigma^2_e$ itself.

## Sequential evaluation

For $[0,1]$-valued per-task quantities $X_t$ with mean $m$ the capital process
$K_t(m)=\prod_{i\le t}(1+\lambda_i(X_i-m))$ is a nonnegative martingale, and the hedged version averages the bets for each sign. By Ville's inequality
$P(\exists t: K_t(m)\ge 1/\alpha)\le\alpha$, so $\{m: K_t(m)<1/\alpha\}$ is a confidence sequence valid at every
stopping time (Waudby-Smith and Ramdas, 2024). The bets $\lambda_i$ are predictable plug-ins. Peeking at a fixed-sample test instead
inflates the false-positive rate.
