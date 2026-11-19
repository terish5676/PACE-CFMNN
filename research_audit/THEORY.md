# Theory for the tough review: statements and proofs for the guide to check

Written 2026-11-16 on branch `tough-review-fixes`. The paper (Claude Doc, rev 575) has Propositions
1–4. This file states and proves the new results the tough review asked for. Each one comes with
the code that implements it and the tests that check it.

**Please check** above all: Proposition 5 (the hardness proof and the two citations it rests on),
and Proposition 6 (clause (iii) was missing from the first version; see its remark).

| # | Result | Review item | Code | Tests | Status |
|---|---|---|---|---|---|
| 5 | Exact PACE-SHAP values are #P-hard | B1 / N1 | `pace_cfmnn/hardness.py` | `tests/test_hardness.py` | proof below; guide to check |
| Cor. 1 | Label-free bound on churn and NFR | B6 part 1 | `audit.churn_bound` | `tests/test_tough_audit.py` | follows from Prop. 4 |
| 6 | Condition S: shrinks that cannot break a correct training prediction | B6 part 2 | `audit.training_safe_shrink` | `tests/test_tough_audit.py`, Run P11 | proof below; clause (iii) added 2026-11-16 |
| 7 | Certificate for PACE-SHAP explanations after an edit | B7 | `audit.pace_shap_certificate` | `tests/test_tough_audit.py`, Run P6 | proof below |
| 8 | Exact single-feature flip sets | B8 / S4 | `pace_cfmnn/flip_exact.py` | `tests/test_tough_explain.py`, Run P7 | proof below |
| 9 | Exact PACE-SHAP for min-membership (GFMM) networks | B2 | `pace_cfmnn/gfmm_explain.py` | `tests/test_gfmm_explain.py`, Run P4 | proof below |
| 10 | Exact Shapley values of the decision (PACE-decision) | B3 / S6 | `pace_cfmnn/decision.py` | `tests/test_tough_explain.py`, Run P5 | proof below |
| – | The recurrence exact mode costs K · 2^d | B9 | `explain.py`, `exact_algo="recurrence"` | `tests/test_tough_explain.py`, Run P8 | a remark, below |

## 0. Notation (as in the paper)

**The model.** A CFMNN has boxes j = 1..H, each with:
- corners v_j ≤ w_j in [0, 1]^d;
- a class c_j;
- a centroid μ_j.

The ramp is f(t) = max(0, 1 − max(0, γ · min(1, t))). The one-feature term is

  m_{j,i}(u) = ½ [ f(u − w_{j,i}) + f(v_{j,i} − u) ],

and box j's membership at y is the mean of these terms:

  b_j(y) = (1/d) Σ_i m_{j,i}(y_i).

The term is 1 inside the box. Outside, it falls linearly with distance, to ½ at distance 1/γ.

**Prediction.**
- Class support: s_c(y) = max over boxes j of class c of b_j(y), rescaled as s'_c = 2 s_c − 1.
- Top group: T(y) = { j : b_j(y) ≥ max_k b_k(y) − τ }, where τ is `tie_tol` (1e-9 by default).
- The winner is the box of T(y) with the smallest squared centroid distance D_j(y) = ‖y − μ_j‖²; a
  remaining tie goes to the lowest index.
- The prediction is the winner's class.

**PACE-SHAP.**
- For the class A predicted at x, the margin is g_A(y) = s'_A(y) − max over c ≠ A of s'_c(y).
- With background rows z^1..z^R and h^r_S = (x on S, z^r elsewhere), the game is
  v(S) = (1/R) Σ_r g_A(h^r_S).
- PACE-SHAP returns this game's Shapley values.
- Proposition 2: b_j(h^r_S) = b_j(z^r) + Σ_{i ∈ S} Δ^r_{j,i}, with Δ^r_{j,i} = [m_{j,i}(x_i) − m_{j,i}(z^r_i)] / d.

---

## Proposition 5 (exact PACE-SHAP values are #P-hard)

**Statement.** Take a CFMNN given by its boxes (2 classes, 4 boxes), a point x and one background
row z. Computing the exact PACE-SHAP values of x is #P-hard, when the box corners are rational
numbers written in binary. It stays hard with τ = 0.

**Source problem.** A weighted voting game has players 1..n with positive integer weights w_i and
a quota q, 1 ≤ q ≤ W = Σ w_i. A coalition S wins when w(S) ≥ q. Player i's Shapley value in this
game is its Shapley–Shubik index SS_i. Computing SS_i is #P-complete when the weights are written
in binary:
- Deng and Papadimitriou (1994), *On the complexity of cooperative solution concepts*,
  Mathematics of Operations Research 19(2): #P-complete under metric reductions;
- Faliszewski and Hemaspaandra (2009), *The complexity of power-index comparison*, Theoretical
  Computer Science 410: many-one #P-completeness.

**Guide, please check** that these two citations say what is claimed here: the volumes, the
years, and the kind of reduction.

**Construction** (`hardness.voting_game_model`):
- d = 2n features: n weight features, then n padding features.
- γ ≥ 2, r = 1/γ, ε = 1 / (4 n w_max).
- The point: x = (½, …, ½).
- The background row: z equals x, except z_i = ½ − t_i on weight feature i, with
  t_i = 2 d ε w_i / γ.
- Four boxes: a1 and a2 of class A, and b1 and b2 of class B.
  - On every weight feature, a1 and b1 span [½, ½ + r], and a2 and b2 span [0, 1].
  - On every padding feature, box j spans [½ + s_j, 1], with s_j = 2 (1 − p_j) / γ.
  - p_{a2} = p_{b2} = ½, p_{a1} = ½ + 2ε (W − q + 1), and p_{b1} = ½ + 2ε (W − q).
- Every number is rational, with size polynomial in the size of the game. All coordinates lie in
  [0, 1]:
  - t_i = w_i / (w_max γ) ≤ r ≤ ½;
  - p ≤ ½ + (W − q + 1) / (2 n w_max) ≤ 1, because q ≥ 1 and W ≤ n w_max;
  - s_j ≤ 1/γ ≤ ½.

**Lemma (memberships).** For every coalition S of features, with K = ¾:
- b_{a2}(h_S) = b_{b2}(h_S) = K;
- b_{a1}(h_S) = K + ε (w(S) − q + 1);
- b_{b1}(h_S) = K + ε (w(S) − q).

Here w(S) sums the weights of the weight features in S.

*Proof.* Take the terms feature by feature.

- **Weight feature i:**
  - x_i = ½ is inside [½, ½ + r], so the term is 1.
  - z_i = ½ − t_i lies t_i ≤ 1/γ below the box, so the term is ½ [1 + (1 − γ t_i)] = 1 − γ t_i / 2.
  - So by Proposition 2, a1 and b1 gain Δ_i = (γ t_i / 2) / d = ε w_i when feature i takes x's
    value.
  - For a2 and b2, both values lie in [0, 1], so the term is 1 either way and Δ = 0.
- **Padding feature:**
  - x and z agree there (both ½), so no box changes with it.
  - ½ lies s_j ≤ 1/γ below box j, so the term is 1 − γ s_j / 2 = p_j.
- **At z:**
  - b_j(z) = (1/d) [Σ_{weight i} term_{j,i}(z_i) + n p_j].
  - For a2 and b2: (1/d)(n + n/2) = ¾ = K.
  - For a1: (1/d)[n − Σ_i γ t_i / 2 + n p_{a1}] = (1/d)[n − d ε W + n/2 + 2 n ε (W − q + 1)].
    Since d = 2n, this is K − ε(q − 1).
  - For b1, in the same way, K − ε q.
- Adding Σ_{i ∈ S} ε w_i to a1 and b1 gives the lemma. ∎

**Proof of the proposition.**

1. Take the class supports.
   - s_A(S) = max(K + ε(w(S) − q + 1), K);
   - s_B(S) = max(K + ε(w(S) − q), K).
2. Compare them. w(S) is an integer.
   - If w(S) ≥ q, both first terms win, and s_A − s_B = ε.
   - If w(S) ≤ q − 1, both maxima equal K.
   - So the margin of class A is g_A(h_S) = 2(s_A − s_B) = 2ε · [w(S) ≥ q].
3. The class predicted at x is A: s_A(x) − s_B(x) = ε > 0, so A wins with τ = 0 (or any τ < ε).
4. With one background row, v(S) = 2ε · [w(S) ≥ q] for every S. This is 2ε times the voting game,
   with the n padding features as null players. Null players do not change the other players'
   Shapley values. So PACE-SHAP's value of weight feature i is exactly 2ε · SS_i.
5. The construction takes polynomial time. One call to an exact PACE-SHAP algorithm, then a
   division by 2ε, gives SS_i. ∎

**Remarks.**
1. **Hardness needs large numbers.** If the weights are bounded by a polynomial in n,
   `shapley_shubik_dp` counts winning coalitions by size and weight in pseudo-polynomial time. So
   the statement is "#P-hard in general", not "hard for every CFMNN". In the construction, the
   corners need about log₂(n w_max) bits.
2. **The model is built, not trained.** Proposition 1 need not hold for it. Whether training can
   produce such a model is not claimed. The result is about the model class (a maximum over boxes
   of mean-membership terms), which is what the paper's §4 sentence "We know of no such result"
   asks about.
3. **Even if B were predicted at x** (for example, a τ larger than ε), the values would be −2ε · SS_i,
   and the reduction still works.
4. **Tests.** `tests/test_hardness.py` builds the construction for random small games and checks
   PACE-SHAP = 2ε × the brute-force index, and that the padding features get 0.
5. **Minimum (GFMM) membership:** no hardness is claimed.

---

## Corollary 1 (a label-free bound on churn; B6, part 1)

**Statement.**
- **Setting.** An edit turns model M into M′. Cert is the set of inputs x where some changed box
  is in T(x) under M or in T′(x) under M′. These are the inputs Proposition 4 does not cover.
- **The bound.** For any distribution D of inputs and any labelling:
  - the share of changed predictions is at most D(Cert);
  - so are the negative flip rate NFR = P_D[M right, M′ wrong] and the change in accuracy.
- **Several edits.** For a run of edits M₀ → … → M_t, the same holds with Cert replaced by the union
  of the edits' certificates.
- **Estimating D(Cert).** Let x_1..x_m be drawn independently from D, independently of how the
  edits were chosen, and let k of them lie in Cert. Then, with probability at least 1 − α,
  D(Cert) ≤ U, where U is the one-sided Clopper–Pearson upper limit for k successes in m trials.
  All three bounds then hold with U.

*Proof.*
- By Proposition 4, a prediction that changes lies in Cert. So the event "changed" is contained in
  the event "in Cert", and NFR and the change in accuracy are each at most the chance of a
  change.
- For several edits, a prediction that differs between M₀ and M_t changes at some step, so it lies
  in that step's certificate.
- The last part is the Clopper–Pearson guarantee for a binomial proportion. ∎

**Remark.** NFR here is normalised by all rows. Normalised by the rows M gets right, divide the
bound by M's accuracy.

---

## Proposition 6 (condition S: shrinks that cannot break a correct training prediction; B6, part 2)

**Statement.**
- **The edit.** It changes only box k, of class c: it shrinks the box (v′_k ≥ v_k and
  w′_k ≤ w_k), or removes it. If the box is kept, its centroid stays the same.
- **Condition S** holds when:
  - (i) no training row of class c lies in the old box but outside the new one;
  - (ii) no training row of class c outside the old box, whose membership in k fell, was won by k
    before the edit;
  - (iii) no training row that M predicts correctly, and that another box j* wins, has
    b_k > b_{j*} there.
- **Conclusion.** Under S, every training row M predicts correctly, M′ also predicts correctly.

*Proof.*

- **Shrinking lowers the membership.** f is non-increasing. Shrinking raises v and lowers w, so
  every term, and hence b_k, can only fall: b′_k ≤ b_k everywhere. Inside the new box, b′_k = 1 = b_k.
  A removed box counts as b′_k = −∞.
- **Case 1: box k won the row** (so the row is of class c).
  - If the row is inside the old box, (i) puts it inside the new box, so b′_k = b_k = 1.
  - If it is outside the old box, b′_k < b_k is ruled out by (ii), so b′_k = b_k.
  - Either way, every membership and every centroid is the same at this row, so the prediction
    is the same.
  - A removal meeting S has no such row: the row would fail (i) or (ii).
- **Case 2: another box j* won the row.**
  - Let μ be the top membership. By (iii), b_k ≤ b_{j*} ≤ μ.
  - So μ is reached by a box other than k: if b_k < μ, by some other box; if b_k = μ, by j*.
  - That box is unchanged, and b′_k ≤ b_k ≤ μ, so the top stays μ.
  - The new top group is the old one, perhaps without k. It still holds j*, whose membership is
    unchanged.
  - Every centroid distance is unchanged (k's centroid too). j* was first, by distance and then by
    index, in the old group. So it is first in the subset, and the prediction is the same. ∎

**Remarks.**
1. **Clause (iii) was missing from the first version** (commit 79c5584). It was found while
   writing this proof, before Run P11 ran.
   - Without it, Case 2 fails when k is strictly on top and j* wins within τ. Lowering the top
     can let a third box into the group, and that box can win on centroid distance.
   - `tests/test_tough_audit.py::test_clause_iii_a_near_tie_lets_a_third_box_in` builds such a
     case (one feature, γ = 1, τ = 0.1). There, (i) and (ii) hold and a correct prediction breaks;
     (iii) catches it.
   - With τ = 0, the winner always has the top membership, and (iii) always holds.
2. The proof does not use Proposition 1.
3. **In practice S rarely holds.** A first shrink at a training error always cuts off a member of
   the box's class (tough review B6, part 3), so it fails (i). Run P11 measures the share.

---

## Proposition 7 (a certificate for PACE-SHAP explanations; B7)

**Statement.**
- **Setting.** An edit turns M into M′, changing the set C of boxes (removed and added boxes
  included). Fix x and the background rows z^1..z^R. Say box j *survives* for (x, z^r) under a
  model if Proposition 3's rules do not prune it.
- **Conditions.**
  - (a) M and M′ predict the same class A at x;
  - (b) no box of C survives for any z^r, under M or under M′.
- **Conclusion.** Then:
  - the exact PACE-SHAP values of x under M and M′ are identical;
  - the sampled values are identical too, when both are computed with the same seed, on the same
    points in the same order.

*Proof.*
- **Under M.** Let U be the unchanged boxes. By Proposition 3, for every S and every r, each
  group's highest membership is reached by a surviving box. Under (b), every surviving box lies in
  U. Its group's boxes in U are a subset of the whole group that contains that box. So the maximum
  over the whole group equals the maximum over the group's boxes in U.
- **Under M′.** The same holds, with the same U and the same memberships: the boxes of U are
  identical in both models.
- **The groups.** Class A versus the rest, by (a); unchanged boxes keep their class.
- **So** g_A(h^r_S) is the same under both models, for every r and S. So v(S) is the same, and so
  are the exact Shapley values.
- **Sampled mode.** The estimate is a fixed function of v on the sampled feature orders. Those
  orders depend only on the seed, d and the point's position, not on the boxes. ∎

**Remark.** The certificate is sound, but in the smoke runs it marked about 100% of the rows. A
changed box survives for some background row at almost every point. Run P6 measures this.

---

## Proposition 8 (exact single-feature flip sets; B8)

**Statement.** Fix x and a feature i, and let u ∈ [0, 1] replace x_i. Then the predicted class is
constant on each open interval between consecutive points of a finite set B. B holds:
- (a) the breakpoints v_{j,i} − 1/γ, v_{j,i}, w_{j,i} and w_{j,i} + 1/γ of every box;
- (b) the points where two memberships differ by exactly 0 or ±τ (on pieces where the difference
  is not constant);
- (c) the points where two boxes' centroid distances are equal.

So evaluating the model at every point of B, and at one point inside each interval, gives the
exact set of values of u that change the decision.

*Proof.*
- **Memberships.** Each term m_{j,i}(u) is piecewise linear, with the breakpoints (a). f is
  linear on [0, 1/γ] and constant elsewhere; for γ ≥ 1 the min(1, t) clause adds no break. The other
  features' terms are fixed. So between consecutive points of (a), every b_j is affine in u.
- **The top group.** The top group is decided by the signs of b_j − b_k + τ over pairs, which are
  affine on each piece. So the group is constant between the points (b).
- **Centroid distances.** D_j(u) = const_j + (u − μ_{j,i})², so D_j − D_k is affine in u. The
  order by distance, ties to the lowest index, is constant between the points (c).
- **So** the winner, and hence the class, is constant on each open interval. ∎

**Remark (pruning).** A box whose highest membership over a piece is below the best lowest
membership minus τ can never enter the top group there, so it can be left out (as in PACE-flip).
The model itself, with every box, gives the class at each evaluation point.

---

## Proposition 9 (exact PACE-SHAP for min-membership networks; B2)

**Statement.** Let a GFMM-family model score box j as b_j(y) = min_i m_{j,i}(y_i), with the margin
taken over class maxima as above. Then the game v(S) for every S can be computed from two
subset-minimum tables per (background row, box), in O(2^d) operations. So PACE-SHAP-min's values
are exact.

*Proof.*
- **The two tables.** For a fixed box and background row, let a_i = m_{j,i}(x_i) and
  c_i = m_{j,i}(z_i). Then b_j(h_S) = min( P(S), Q(N \ S) ), where:
  - P(S) = min_{i ∈ S} a_i;
  - Q(T) = min_{i ∈ T} c_i;
  - an empty minimum is +∞.
- **Building them.** Each table takes O(2^d) operations, by P(S) = min(P(S without its lowest
  feature), a_lowest).
- **The game.** Class maxima, the margin and the mean over background rows are taken over these
  exact values for every S.
- **The values.** The Shapley values then come from the 2^d values of v. ∎

**Remark (pruning).** For every S, b_j(h_S) lies between a floor and a ceiling:
- the floor is min_i min(a_i, c_i);
- the ceiling is min_i max(a_i, c_i).

So a box whose ceiling is below the highest floor in its group is never its group's maximum.
`PaceShapMin` drops such boxes, as Proposition 3 does for the mean. Run P4 checks the code against
the brute force on hyperbox-brain models.

---

## Proposition 10 (exact Shapley values of the decision; B3)

**Statement.** Let v(S) = (1/R) Σ_r 1[ M(h^r_S) = A ], the share of background rows for which the
mixed input keeps the class predicted at x. Here M is the full rule: the top group with τ, then the
nearest centroid, then the lowest index. Then v(S) for every S is computed exactly from two
subset-sum tables per (background row, box).

*Proof.*
- **Memberships.** b_j(h^r_S) = b_j(z^r) + Σ_{i ∈ S} Δ^r_{j,i}, by Proposition 2.
- **Centroid distances.** D_j(h^r_S) = D_j(z^r) + Σ_{i ∈ S} δ^r_{j,i}, with
  δ^r_{j,i} = (x_i − μ_{j,i})² − (z^r_i − μ_{j,i})².
- **The decision.** For each S and r, the decision uses only these two numbers per box, so it is
  computed exactly.
- **The values.** v(S) is exact, and so are its Shapley values. ∎

**Remark.**
- At a tie the margin is 0, and the decision comes from the centroid rule. This game credits the
  features that keep or lose the class there.
- Run P5 compares this game's values with PACE-SHAP's at tied points.
- The tests check it against `brute_force_decision` (the model's prediction on every mixed input)
  to 1e-12.

---

## Remark (the recurrence exact mode; B9)

For a kept (background row, box) pair, the subset sums of the Δ_{j,i} are built in O(2^d) by
Σ(S) = Σ(S without its lowest feature) + Δ_lowest. The matrix product costs O(d · 2^d). So a point
costs K · 2^d rather than K · d · 2^d. The two modes add the same numbers in different orders, so
they agree up to rounding, not to the last bit. Run P8 checks agreement within 1e-12, and with shap's
ExactExplainer within 1e-9.
