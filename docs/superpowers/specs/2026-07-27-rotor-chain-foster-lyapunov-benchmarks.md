# Benchmark Problems for Symbolic Foster–Lyapunov Discovery in Boundary-Thermostatted Rotor Chains

Dated 2026-07-27. Transcribed from the benchmark draft into repo docs on 2026-07-28.
These four problems are the next-stage targets for the discovery pipeline: instead of
deterministic Lyapunov functions for ODEs, the generator must produce **stochastic
Foster–Lyapunov certificates** for hypoelliptic Langevin dynamics on `T^N × R^N`.

> Transcription note: recovered from a PDF text extraction; all formulas were
> reconstructed. One spot marked ⚠ below was ambiguous in the source and should be
> checked against reference [2] before implementing the falsifier.

## 1 Common setup

### 1.1 Hamiltonian rotor chain

For $N \ge 2$, let $q=(q_1,\dots,q_N)\in\mathbb{T}^N$, $p=(p_1,\dots,p_N)\in\mathbb{R}^N$.
Here $\mathbb{T}=\mathbb{R}/(2\pi\mathbb{Z})$ is the one-dimensional torus: every $q_i$
is understood modulo $2\pi$; $p_i$ is its conjugate momentum. Bond variables:

$$r_i = q_{i+1}-q_i \ (\mathrm{mod}\ 2\pi), \qquad i=1,\dots,N-1,$$

and the Hamiltonian

$$H_N(q,p) = \tfrac12\sum_{i=1}^{N} p_i^2 + \sum_{i=1}^{N-1} W_i(r_i). \tag{1}$$

The general theory allows smooth, periodic, non-degenerate interactions. To make the four
benchmarks directly comparable we fix

$$W_i(r)=W(r)=1-\cos r, \qquad w(r)=W'(r)=\sin r. \tag{2}$$

Adding the constant 1 to $-\cos r$ does not change the dynamics but makes $H_N \ge 0$.

### 1.2 Endpoint Langevin baths (N ≥ 3)

Only rotors $1$ and $N$ are coupled to heat baths. Concrete parameters:

$$\gamma_1=\gamma_N=1,\qquad T_1=1,\ T_N=2, \tag{3}$$

zero external torques. Dynamics:

$$
\begin{aligned}
dq_i &= p_i\,dt, && i=1,\dots,N, & \text{(4a)}\\
dp_1 &= \big(w(r_1)-p_1\big)\,dt+\sqrt{2}\,dB^1_t, && & \text{(4b)}\\
dp_i &= \big(w(r_i)-w(r_{i-1})\big)\,dt, && i=2,\dots,N-1, & \text{(4c)}\\
dp_N &= \big(-w(r_{N-1})-p_N\big)\,dt+2\,dB^N_t, && & \text{(4d)}
\end{aligned}
$$

with $B^1, B^N$ independent standard Brownian motions. In general the noise coefficient at
endpoint $b$ is $\sqrt{2\gamma_b T_b}$ (hence $\sqrt2$ at $T_1=1$ and $2$ at $T_N=2$).

Let $F_i(q) = -\partial_{q_i} H_N(q,p)$. The infinitesimal generator on a $C^2$ function $f$:

$$\mathcal{L}_N f = \sum_{i=1}^{N}\big(p_i\partial_{q_i} f + F_i(q)\partial_{p_i} f\big)
 - p_1\partial_{p_1} f - p_N\partial_{p_N} f
 + \partial^2_{p_1 p_1} f + 2\,\partial^2_{p_N p_N} f. \tag{5}$$

In particular

$$\mathcal{L}_N H_N = (1-p_1^2) + (2-p_N^2). \tag{6}$$

Thus $H_N$ alone does not detect energy trapped in the undamped interior rotors and is
**not** a sufficient Foster–Lyapunov function.

### 1.3 One-bath N = 2 specialization

The baby benchmark is deliberately *not* obtained by identifying the two endpoints in (4).
Only rotor 1 is thermostatted ($\gamma_1=T_1=1$); rotor 2 has neither friction nor noise:

$$
\begin{aligned}
dq_i &= p_i\,dt, \quad i=1,2, & \text{(7a)}\\
dp_1 &= \big(w(r_1)-p_1\big)\,dt+\sqrt2\,dB^1_t, & \text{(7b)}\\
dp_2 &= -w(r_1)\,dt. & \text{(7c)}
\end{aligned}
$$

Generator:

$$\mathcal{L}_2 f = p_1\partial_{q_1}f + p_2\partial_{q_2}f
 + w(r_1)(\partial_{p_1}-\partial_{p_2})f - p_1\partial_{p_1}f + \partial^2_{p_1p_1}f, \tag{8}$$

and

$$\mathcal{L}_2 H_2 = 1-p_1^2. \tag{9}$$

The unthermostatted second rotor creates the same fast-rotor obstruction as an interior
rotor in a longer chain.

### 1.4 Target certificate

For $R>0$ let $K_R=\{(q,p): H_N(q,p)\le R\}$. The model must generate an explicit,
periodic-in-$q$, $C^2$ function $V_N:\mathbb{T}^N\times\mathbb{R}^N\to[1,\infty)$ with
compact sublevel sets, together with constants $c,C,R>0$, such that

$$\mathcal{L}_N V_N \le C\,\mathbf{1}_{K_R} - \varphi_N(V_N)
 \quad\text{on } \mathbb{T}^N\times\mathbb{R}^N. \tag{10}$$

Parametric drift family:

$$\varphi_{\alpha,c}(v) = c\,\frac{v}{(\alpha+1+\log v)^\alpha},\qquad v\ge 1. \tag{11}$$

The offset $\alpha+1$ makes this increasing and concave on $[1,\infty)$; its precise value
is inessential. The significant feature is the large-$v$ behavior $v/(\log v)^\alpha$.

**These are stochastic Foster–Lyapunov problems.** The reward and falsification code must
evaluate the **full generator** (8) or (5), including the second derivatives in every noisy
momentum (one for $N=2$, two for $N\ge3$). Replacing $\mathcal{L}_N V$ by the first-order
Lie derivative would define a different problem.

| N | state dim | status | target α | associated rate |
|---|---|---|---|---|
| 2 | 4 | proved | 1 | $\exp(-ct^{1/2})$ |
| 3 | 6 | proved | 1 | $\exp(-ct^{1/2})$ |
| 4 | 8 | proved | 1 | $\exp(-ct^{1/2})$ |
| 5 | 10 | conjectural | 3 | $\exp(-ct^{1/4})$ |

## 2 The four testing problems

### Testing problem 1 (N = 2: one fast unthermostatted rotor)

Use the one-bath dynamics (7). Explicitly

$$H_2=\tfrac12(p_1^2+p_2^2)+(1-\cos(q_2-q_1)),\qquad
dp_1=(\sin(q_2-q_1)-p_1)dt+\sqrt2\,dB^1_t,\qquad dp_2=-\sin(q_2-q_1)\,dt.$$

Find a globally $C^2$, coercive $V_2$ and $c,C,R>0$ with

$$\mathcal{L}_2 V_2 \le C\,\mathbf{1}_{K_R} - c\,\frac{V_2}{2+\log V_2}. \tag{12}$$

The output must be periodic in $(q_1,q_2)$ and coercive in $(p_1,p_2)$, **including along
the fast unthermostatted direction** $|p_2|\to\infty$ with $p_1$ bounded.

**Known reference information.** Cuneo and Poquet [1] study precisely this two-rotor,
one-bath model and prove the stretched-exponential scale $\exp(-ct^{1/2})$ cannot be
improved. The matching upper bound follows from the three-rotor Foster–Lyapunov proof by
omitting the third rotor [2]. The one-branch reference construction, for sufficiently
small $0<\theta<1$:

$$V^{\mathrm{ref}}_2 = 1 + e^{\theta H_2}
 + A\,\rho_2(p)\,\tilde p_2^{\,2}\exp\!\big(\tfrac{\theta}{2}\tilde p_2^{\,2}\big). \tag{13}$$

Gate, fixed explicitly:

$$b(u)=\begin{cases}0,&u\le0,\\ \exp(-1/u),&u>0,\end{cases}
\qquad
\chi(z)=\frac{b(z-1)}{b(z-1)+b(4-z)}, \tag{14}$$

so $\chi=0$ for $z\le1$ and $\chi=1$ for $z\ge4$. Taking $k=2$ and $R_\rho=4$:

$$S_2(p_1)=p_1^4+4,\qquad \rho_2(p_1,p_2)=\chi\!\Big(\frac{p_2^2}{S_2(p_1)^2}\Big). \tag{15}$$

In particular $\rho_2=0$ when $|p_2|\le p_1^4+4$ and $\rho_2=1$ when $|p_2|\ge 2(p_1^4+4)$.
Regularized inverse powers:

$$I_j(p_1,p_2)=\begin{cases}\rho_2(p_1,p_2)/p_2^{\,j},&p_2\neq0,\\ 0,&p_2=0,\end{cases}
\qquad j=1,\dots,4, \tag{16}$$

each globally $C^\infty$ since $\rho_2$ vanishes near $p_2=0$. The corrected fast momentum
(averaging the oscillatory interaction force; concrete normal form for the cosine
benchmark):

$$
\begin{aligned}
\tilde p_2 = p_2 &- (\cos r_1)\,I_1 - (p_1\cos r_1)\,I_2
 - (p_1^2\cos r_1 + p_1\sin r_1 + \cos^2 r_1)\,I_3\\
&+ \big(-p_1^3\cos r_1 - 3p_1^2\sin r_1 - 3p_1\cos^2 r_1 + p_1\cos r_1
 - \tfrac12\sin r_1\cos r_1 + 2\sin r_1\big)\,I_4.
\end{aligned} \tag{17}
$$

On the fully selected fast branch $\rho_2=1$ this agrees with the inverse-power normal
form and satisfies

$$d\tilde p_2 = \Big(-\frac{1}{2\tilde p_2^{\,3}} + O_f(\tilde p_2^{-4})\Big)dt
 + \Big(-\frac{\sqrt2\cos r_1}{\tilde p_2^{\,2}} + O_f(\tilde p_2^{-3})\Big)dB^1_t. \tag{18}$$

$O_f$ denotes weighted fast-region remainders: coefficients may depend polynomially on the
thermostatted endpoint momenta but are controlled by the gate. There are $a_1,a_2>0$ with

$$1 + a_1 e^{\theta H_2} \le V^{\mathrm{ref}}_2 \le a_2(1+p_2^2)e^{\theta H_2}. \tag{19}$$

The separate test function constructed in [1] establishes the *lower* relaxation bound; it
has positive upper drift and must not be confused with the negative-drift reference (13).

### Testing problem 2 (N = 3: one fast interior rotor)

Set $N=3$ in (1)–(4). Explicitly

$$H_3=\tfrac12(p_1^2+p_2^2+p_3^2)+(1-\cos(q_2-q_1))+(1-\cos(q_3-q_2)),$$
$$dp_1=(\sin(q_2-q_1)-p_1)dt+\sqrt2\,dB^1_t,\quad
dp_2=(\sin(q_3-q_2)-\sin(q_2-q_1))dt,\quad
dp_3=(-\sin(q_3-q_2)-p_3)dt+2\,dB^3_t.$$

Find $V_3$ and $c,C,R>0$ with

$$\mathcal{L}_3 V_3 \le C\,\mathbf{1}_{K_R} - c\,\frac{V_3}{2+\log V_3}. \tag{20}$$

Output must be coercive in $(p_1,p_2,p_3)$ and globally $C^2$.

**Known reference information.** Rigorous for sufficiently small $\theta>0$ [2]:

$$V^{\mathrm{ref}}_3 = 1 + e^{\theta H_3}
 + A\,\rho_3(p)\,\tilde p_2^{\,2}\exp\!\big(\tfrac{\theta}{2}\tilde p_2^{\,2}\big), \tag{21}$$

where $\rho_3$ is a smooth gate selecting the regime where $|p_2|$ dominates the endpoint
momenta and $\tilde p_2$ is a corrected fast momentum. First two counterterms:

$$\tilde p_2 = p_2 - \frac{\cos r_1+\cos r_2}{p_2}
 - \frac{p_1\cos r_1 + p_3\cos r_2}{p_2^{\,2}} + O_f(p_2^{-3}). \tag{22}$$

Higher counterterms are selected recursively so that on the gated fast branch

$$d\tilde p_2 = \Big(-\frac{1}{\tilde p_2^{\,3}} + O_f(\tilde p_2^{-4})\Big)dt
 + \Big(-\frac{\sqrt2\cos r_1}{\tilde p_2^{\,2}} + O_f(\tilde p_2^{-3})\Big)dB^1_t
 + \Big(-\frac{2\cos r_2}{\tilde p_2^{\,2}} + O_f(\tilde p_2^{-3})\Big)dB^3_t. \tag{23}$$

> ⚠ resolved 2026-08-04: a second verbatim extraction supplied by the user
> (2026-07-27-rotor-fl-benchmarks-original-extraction.txt) reads "− 1 p 3 2" here,
> confirming $-1/\tilde p_2^{\,3}$ for $N=3$ (vs $-1/(2\tilde p_2^{\,3})$ in (18) for
> $N=2$). Cross-checking against [2] remains a nice-to-have, no longer a blocker.

As in $N=2$, the fast-branch expression is smoothly extended where $\rho_3=0$. Constants
$a_1,a_2>0$:

$$1 + a_1 e^{\theta H_3} \le V^{\mathrm{ref}}_3 \le a_2(1+p_2^2)e^{\theta H_3}. \tag{24}$$

For a strict blind test, (21)–(24) should be withheld from the generator and used only
for comparison.

### Testing problem 3 (N = 4: two fast interior rotors and resonances)

Set $N=4$. Interior momentum equations

$$dp_2=(\sin(q_3-q_2)-\sin(q_2-q_1))dt,\qquad dp_3=(\sin(q_4-q_3)-\sin(q_3-q_2))dt,$$

endpoints per (4). Find a globally $C^2$, coercive $V_4$ and $c,C,R>0$ with

$$\mathcal{L}_4 V_4 \le C\,\mathbf{1}_{K_R} - c\,\frac{V_4}{2+\log V_4}. \tag{25}$$

The certificate must cover **all high-energy regimes**: only $p_2$ fast, only $p_3$ fast,
both fast and non-resonant, and both fast and resonant.

**Known reference information.** Rigorous [3]. For any
$0<\theta<\min(T_1^{-1},T_4^{-1})=\tfrac12$, the reference patches regime-specific pieces:

$$V^{\mathrm{ref}}_4 = 1 + e^{\theta H_4} + \psi_2(p)\tilde V_2 + \psi_3(p)\tilde V_3
 + M\,\psi_c(p)\tilde V_c, \tag{26}$$

with smooth regime gates $\psi_2,\psi_3,\psi_c$. One-fast-rotor pieces:

$$\tilde V_j \asymp \exp\!\big(|\tilde p_j|^a + \tfrac{\theta}{2}\tilde p_j^{\,2}\big),
\qquad j=2,3,\quad 0<a<1,$$

both-fast piece:

$$\tilde V_c \asymp \tilde H_c\,e^{\theta\tilde H_c},\qquad
\tilde H_c \asymp \tfrac12(\tilde p_2^{\,2}+\tilde p_3^{\,2}).$$

Tildes denote averaging and resonance counterterms, not merely a change of notation.
Globally

$$1 + e^{\theta H_4} \le V^{\mathrm{ref}}_4
 \le a_3\big(e^{|\tilde p_2|^a}+e^{|\tilde p_3|^a}\big)e^{\theta H_4} \tag{27}$$

for some $a_3>0$. The reference structure can be supplied as a grammar prior or withheld
for a genuinely blind test.

### Testing problem 4 (N = 5: conjectural central-rotor bottleneck)

Set $N=5$; interior dynamics

$$dp_2=(\sin(q_3-q_2)-\sin(q_2-q_1))dt,\quad
dp_3=(\sin(q_4-q_3)-\sin(q_3-q_2))dt,\quad
dp_4=(\sin(q_5-q_4)-\sin(q_4-q_3))dt.$$

**Find or falsify** the existence of a globally $C^2$, coercive $V_5$ and $c,C,R>0$ with

$$\mathcal{L}_5 V_5 \le C\,\mathbf{1}_{K_R} - c\,\frac{V_5}{(4+\log V_5)^3}. \tag{28}$$

"Find or falsify" is important: failure of a particular symbolic ansatz is not a disproof,
but an explicit sequence of states along which its drift inequality fails is a useful
counterexample for further training.

**Conjecture 1 (benchmark conjecture for N = 5).** There are $\theta,c,C,R>0$ and a smooth
$V_5\ge1$ with $\log V_5 = \theta H_5 + o(H_5)$ as $H_5\to\infty$ such that (28) holds.

The exponent 3 is inferred from the conjectured mixing law in [3, Remark 5.3]. If
$\varphi(v)\asymp v/(\log v)^\alpha$ then
$H_\varphi(u)=\int_1^u ds/\varphi(s) \asymp (\log u)^{\alpha+1}$, and the corresponding
stretched-exponential power is $1/(\alpha+1)$. The prediction

$$k_N = \frac{1}{2\lceil N/2\rceil - 2}$$

gives $k_5=1/4$, hence $\alpha=3$.

Heuristic starting point:

$$V^{\mathrm{ansatz}}_5 = 1 + e^{\theta H_5}
 + A\,\psi_3(p)\,\tilde p_3^{\,6}\exp\!\big(\tfrac{\theta}{2}\tilde p_3^{\,2}\big)
 + \text{corrections for the other fast and resonant regimes}, \tag{29}$$

motivated by the unproved effective scaling

$$\mathcal{L}_5\tilde p_3 \sim -\kappa\,\tilde p_3^{-7}
\ \Longrightarrow\
\frac{\mathcal{L}_5 V_5}{V_5} \sim -\tilde p_3^{-6} \sim -(\log V_5)^{-3}.$$

Both the scaling and the required corrections are conjectural. A serious search must also
treat high energy shared among $(p_2,p_3)$, $(p_3,p_4)$, or $(p_2,p_3,p_4)$, where new
resonances can arise.

## 3 Suggested evaluation protocol

For each $N$, the model receives $H_N$, $\mathcal{L}_N$, the target exponent $\alpha$, and
an allowed symbolic grammar. It returns an expression tree for $V_N$ plus numerical or
symbolic values for free parameters. Evaluation stages:

1. **Syntax and regularity.** Expression is real, periodic in each angle, globally
   defined, $C^2$. Denominators like $1/p_i$ require smooth cutoffs vanishing near
   $p_i=0$.
2. **Positivity and coercivity.** Search for violations of $V_N\ge1$; verify growth as
   $\|p\|\to\infty$.
3. **Generator falsification.** Maximize
   $\mathcal{L}_N V_N + c\,V_N/(\alpha+1+\log V_N)^\alpha$ outside $K_R$, with adversarial
   sampling concentrated in fast-rotor and near-resonant regimes.
4. **Tail certification.** A bounded-box numerical check is not a global proof. Split
   momentum space into asymptotic regimes and prove the sign of the leading terms in each.
5. **Blind versus guided scores.** Run a blind task using only the dynamics, then a guided
   task where the structural reference forms (13)/(21)/(26)/(29) are included in the
   grammar prior. Report both scores.

A practical first grammar: arithmetic, integer and learned real constants, `sin`, `cos`,
`exp`, `log`, (smooth) absolute value, and a fixed smooth cutoff primitive. The $N=4$ and
$N=5$ problems are unlikely to admit short global expressions without some mechanism for
regime-dependent patching.

## 4 Questions left open in this draft

1. Fully global certificate only, or also a finite-domain discovery track compatible with
   existing numerical falsifiers?
2. Bath temperatures/frictions fixed as above, or sampled during training to test
   parameter generalization (separately for one-bath $N=2$ and two-bath $N\ge3$)?
3. For $N=5$: supply $\alpha=3$, or have the model jointly search over $\alpha$ and $V_5$?
4. Which smooth cutoff functions and piecewise constructions should be atomic tokens in
   the discrete diffusion vocabulary?

## References

1. N. Cuneo, C. Poquet. *On the relaxation rate of short chains of rotors interacting with
   Langevin thermostats.* Electron. Commun. Probab. 22 (2017), no. 35. doi:10.1214/17-ECP62
2. N. Cuneo, J.-P. Eckmann, C. Poquet. *Non-equilibrium steady state and subgeometric
   ergodicity for a chain of three coupled rotors.* Nonlinearity 28(7):2397–2421, 2015.
   doi:10.1088/0951-7715/28/7/2397
3. N. Cuneo, J.-P. Eckmann. *Non-equilibrium steady states for chains of four rotors.*
   Commun. Math. Phys. 345(1):185–221, 2016. doi:10.1007/s00220-015-2550-2
4. H. Zou, J. Feng, H. Zhao, Y. Shi. *Analytical Lyapunov function discovery: an RL-based
   generative approach.* ICML 2025, PMLR 267:80776–80804.
   https://proceedings.mlr.press/v267/zou25a.html
