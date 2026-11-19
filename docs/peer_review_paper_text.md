# Draft paper text for the peer-review fixes (Phase 1)

Text to go into the paper (Claude Doc, currently rev 326). Nothing here is in the paper yet.
Every number comes from `results/peer_review/reanalysis.md` (post hoc; written by
`scripts/peer_review_reanalysis.py`) or from the code lines named beside it.

Parts:
- A. New §3: model, notation and four propositions (review items B1, B2, D2).
- B. Sentences to replace (A5, C1, C6, D4).
- C. New result paragraphs (A4, A8, A11, C2, C3, C5, C7).

---

## A. Section 3: the model and its properties

### 3.1 Notation

All features are scaled to [0, 1]; a point is x = (x_1, …, x_d). Box j has a lower corner v_j, an
upper corner w_j (v_j ≤ w_j on every feature), a class c_j and a centroid μ_j, the mean of the
training rows the box holds. H is the number of boxes.

**Ramp.** For a sensitivity γ > 0 (we use γ = 4),

    f(t) = max(0, 1 − max(0, γ · min(1, t))).

So f(t) = 1 for t ≤ 0, falls linearly to 0 at t = 1/γ, and stays 0.

**One-feature term.** The score of box j on feature i is

    m_{j,i}(x_i) = ½ · [ f(x_i − w_{j,i}) + f(v_{j,i} − x_i) ].

It is 1 exactly when v_{j,i} ≤ x_i ≤ w_{j,i}. Outside the box on that feature, at distance t, it is
1 − γt/2, and it stays at ½ once t ≥ 1/γ (0.25 for γ = 4).

**Membership.** b_j(x) = (1/d) · Σ_i m_{j,i}(x_i). It lies in [½, 1], and b_j(x) = 1 exactly when x
is inside box j.

**Prediction.** Let T(x) = { j : b_j(x) ≥ max_k b_k(x) − τ } be the tied top group, with a small
tolerance τ (1e-9). The winner is the box of T(x) whose centroid is nearest to x (lowest index if
two are equally near), and the prediction is its class.

**Margin.** The support of class c is s_c(x) = max over boxes of class c of b_j(x). The margin of
class A is

    g_A(x) = 2 · [ s_A(x) − max_{c ≠ A} s_c(x) ].

The margin does not use the centroids. When boxes of A and of another class tie at the top,
g_A(x) = 0 and the class is decided by the nearest-centroid rule alone.

**Training.** One pass over the rows. For a row (x, y): if no box of class y exists, create the
point box v = w = x. Otherwise take the boxes of class y; if x is inside the best one, add x to its
centroid; else grow the best box that stays under the size limit Σ_i (w_i − v_i) ≤ d·θ so that it
holds x; if none can grow, create a point box. Boxes are never shrunk and never removed.

(Code: `pace_cfmnn/cfmnn_core.py:38–49` for f and m; `:170–208` for training; `:367–373` for the
tie rule. Margin: `pace_cfmnn/explain.py:394`.)

### 3.2 Properties

**Proposition 1 (training errors are ties).** After training on rows (x_r, y_r):

1. every training row is inside at least one box of its own class, so max_k b_k(x_r) = 1;
2. if the prediction at x_r is wrong, the winning box k has a class other than y_r and
   b_k(x_r) ≥ 1 − τ. With τ = 0, x_r is inside box k.

So every training error is a tie between a box of the right class and a box of the wrong class,
both holding the row, which the nearest-centroid rule breaks the wrong way.

*Proof.* (1) When row r is processed, it is already inside a box of class y_r, or a box of class
y_r is grown to hold it, or a point box is created at x_r. In each case x_r is inside a box of class
y_r at that moment. Later steps only grow boxes or add boxes, so it stays inside. Inside a box
every one-feature term is 1, so the membership is 1, the largest value possible. (2) The winner is
in T(x_r), so its membership is at least 1 − τ. If the prediction is wrong, its class is not y_r. ∎

*Remarks.* The proposition holds for every θ, γ and row order, for both centroid rules and both
expansion rules; it uses only the fact that training never shrinks a box. With τ > 0, part 2 says
x_r is within 2dτ/γ of box k on every feature. Simpson's network avoids such overlaps with an
overlap test and a contraction step; CFMNN has neither, and this is the price. In all 347,339
training errors of the 520 models of Section 7 the property held, which checks our code. The
proposition is about training rows only. It stops holding once a box is shrunk (Section 6), because
a shrink drops the rows it cuts off.

**Proposition 2 (coalition values without the model).** Take a point x, a background row z and a
set S of features. Let h be the point with x's values on S and z's values elsewhere. Then for every
box j

    b_j(h) = b_j(z) + Σ_{i ∈ S} Δ_{j,i},   Δ_{j,i} = [ m_{j,i}(x_i) − m_{j,i}(z_i) ] / d.

*Proof.* b_j is a mean of one-feature terms, and h differs from z only on S. ∎

*Consequence.* Fix the class A predicted at x and background rows z_1, …, z_R. The game

    v(S) = (1/R) · Σ_r g_A(x_S, z_{r, −S})

can be evaluated for any S from the table Δ, with no call to the model. For d ≤ 12 PACE-SHAP
evaluates v on all 2^d sets and applies the Shapley formula

    φ_i = Σ_{S ⊆ N∖{i}} [ |S|! (d − |S| − 1)! / d! ] · [ v(S ∪ {i}) − v(S) ],

so its values are the exact Shapley values of this game. The cost is O(2^d · d · R · H) operations
before pruning. Three things are fixed by this statement and should be read with every use of
"exact": the game is interventional (features outside S come from background rows, independently of
x); it depends on the chosen background rows; and it explains the margin g_A, not the
nearest-centroid rule. The values add up to g_A(x) − (1/R) Σ_r g_A(z_r); for a tied point
g_A(x) = 0. This is enumeration made cheap. It is not a polynomial-time algorithm in d, as TreeSHAP
is for trees.

**Proposition 3 (pruning changes nothing).** Split the boxes into two groups: class A and all other
classes. For a fixed x and z, write lo_j = b_j(z) + Σ_i min(0, Δ_{j,i}) and
hi_j = b_j(z) + Σ_i max(0, Δ_{j,i}). A box j can be dropped from its group without changing v(S) for
any S if either

1. hi_j < max over its group of lo_k, or
2. some box k of its group has b_k(z) − b_j(z) + Σ_i min(0, Δ_{k,i} − Δ_{j,i}) > 0.

*Proof.* By Proposition 2, b_j(h) lies in [lo_j, hi_j] for every S, so under (1) another box of the
group is above j for every S. Under (2), b_k(h) − b_j(h) = b_k(z) − b_j(z) + Σ_{i∈S} (Δ_{k,i} −
Δ_{j,i}), which is at least the left side of (2), so k is above j for every S. The margin uses only
each group's maximum. ∎ (Code: `pace_cfmnn/explain.py:658–700`; the code keeps a 1e-12 guard so
rounding never removes a maximum.)

**Proposition 4 (certificates).** Let an edit turn model M into M′ and let C be the set of boxes
whose corners, centroid or class differ between them, including boxes removed or added. If no box
of C is in T(x) under M and none is in T′(x) under M′, the prediction at x is the same. The same
holds for the smaller set that also leaves out every x whose T(x) and T′(x) hold boxes of one class
only.

*Proof.* If no changed box is in T(x), the top membership under M is reached by an unchanged box,
and likewise under M′. Unchanged boxes have the same membership in both models, so the two top
values are equal, and T(x) and T′(x) hold the same boxes. Their centroids and their order are
unchanged, so the same box wins. For the smaller set: if every box in both groups has class c, the
prediction is c both times. ∎ Both models must use the same τ. A row added inside a box changes only
that box's centroid, and the box then counts as changed. (Code: `pace_cfmnn/audit.py:24–44`,
`:77–85`.)

---

## B. Sentences to replace

| Where | Now | Replace with |
|---|---|---|
| Abstract | "We find why it errs and fix it safely. On all 13 datasets, every training error of CFMNN we checked is a tie between a box of the right class and a box of the wrong class" | "We show why it makes training errors. Because training never shrinks a box, every training error is a tie between a box of the right class and a box of the wrong class (Proposition 1)" |
| Abstract | "on 3 of 13 datasets, level on the other 10 and worse on none, as expected when both target the same values" | "on 3 of 13 datasets and equivalent, within a tenth of KernelSHAP's own lead over a random ranking, on 12 of 13" |
| §1, contribution 2 | "the finding that every CFMNN training error we checked is a tie" | "a proof that every CFMNN training error is a tie" |
| §6, first paragraph | "This held for 100% of training errors on all 6 datasets … over 20 seeds." | "This follows from how CFMNN trains (Proposition 1). It held for all 347,339 training errors of the 520 models in Section 7, which checks our code." |
| §6, after the first paragraph | (new) | "The proposition covers training errors only. On S1–S3, Wine and Digits the model makes no training errors at θ = 0.3, so self-repair has nothing to fix there, and their test errors have another cause: with 2.6–4.9 training rows per box the model is close to memorising its training rows." |
| §7 Setup | (new, once) | "Two sets of 13 datasets appear below. The accuracy and repair tests use S1–S3, Wine, Breast Cancer, Digits, Thyroid, Shuttle, Skin, MAGIC, Letter, Landsat and Page Blocks. The conditional-replacement run uses the 11 explanation datasets plus S1c and S2c, which are S1 and S2 with feature 3 made a noisy copy of feature 0." |
| §7 Setup | (new, once) | "With 10 seeds the smallest two-sided Wilcoxon p is 0.002. After Holm correction across 11 datasets, a significant result needs at least 9 of 10 seeds to agree, and across 13 datasets all 10. So we give each difference with a 95% interval, and we call two methods equivalent only when an equivalence test says so." |
| Everywhere | "level" | "not significantly different", or "equivalent" where the equivalence test supports it |

---

## C. New result paragraphs

**Ties decide many predictions (§7, before Faithfulness).** On the base model (seeds 0–4, every
test row), the share of test rows whose class is decided by the nearest-centroid rule is 0% on
S1–S3 and Wine, under 1% on Breast Cancer and Digits, 5% on Thyroid, 13% on Shuttle, 20% on Skin,
25% on Landsat, 57% on Letter, 73% on Page Blocks and 76% on MAGIC. Tied rows are also where the
model errs: they hold 90–100% of the test errors on Skin, MAGIC, Letter and Page Blocks, 83% on
Shuttle, and 53–60% on Landsat and Thyroid. On a tied row PACE-SHAP's values add up to minus the
average background margin: they say how the features bring the two classes level, not why one of
them was then chosen. (reanalysis §10)

**Equivalence, like for like (§7 Faithfulness).** With the same 200 rows and every feature,
PACE-SHAP and KernelSHAP were equivalent on 12 of 13 datasets, by deletion and by insertion (margin:
10% of KernelSHAP's lead over a random ranking). The exception is Landsat, where PACE-SHAP was more
faithful (difference −0.045, 95% interval −0.054 to −0.036). Against KernelSHAP's usual setting they
were equivalent on 6 of 11 by deletion. (reanalysis §1)

**The synthetic sets as one unit (§7 Faithfulness).** Counting the synthetic sets as one unit, as
the trust-signal tests do, PACE-SHAP was more faithful than KernelSHAP's usual setting on 6 of 9
units by deletion and by insertion, and like for like on 3 of 9 by deletion (the synthetic family,
Breast Cancer, Landsat) and 2 of 9 by insertion. Across the 9 units, a signed-rank test on the mean
differences gave p = 0.02 for the usual setting (both tests), and like for like p = 0.20 by deletion
and p = 0.004 by insertion, where all 9 small differences favour PACE-SHAP (median 0.0015).
(reanalysis §2, §3)

**The other PACE explanations (§7 Faithfulness, after the table).** The contrastive explanation,
the score breakdown and PACE-multi answer other questions than "which features carry the margin",
and they rank features for deletion worse than KernelSHAP: the contrastive explanation on 9 of 11
datasets, the score breakdown on all 11, PACE-multi on 6. All beat a random ranking, except the
score breakdown on 2 datasets, where the difference is not significant. We add their columns to the
table. (reanalysis §6)

**Repair, with intervals (§7 Repair, second run).** Self-repair's gain in macro-F1 was 6.0 points on
Thyroid (95% interval 2.6 to 9.5), 1.5 on Landsat (0.2 to 2.8), 2.6 on Page Blocks (0.8 to 4.4),
3.3 on MAGIC (2.2 to 4.4), 2.3 on Letter (1.8 to 2.7) and 5.7 on Skin (3.7 to 7.6). Self-repair and
active learning were not shown to be equivalent within 1 point on any dataset. (reanalysis §4)

**Fix share and accuracy (§6 and §7).** Fixed minus broken is the change in correct predictions, so
an update that leaves accuracy unchanged has a fix share near 50% whatever it does. Fix share
therefore mostly restates the accuracy result. The share of test predictions an update breaks does
not have this problem; we report it beside fix share. (reanalysis §5)

**Counterfactuals under a refit (§7 Follow-up tests).** When the model is refitted in another row
order, a PACE counterfactual still gives the target class in 17–55% of refits on 10 of the 11
datasets (98% on Shuttle), and a PACE-sparse one in 24–42% (98% on Shuttle). DiCE's random search
holds in 28–59% (77% on Shuttle), its genetic search in 52–99%, and the nearest training point of
the target class in 32–100%. (reanalysis §7)

**Cost of the trust signals (§7 Trust signals).** Computing PD, EI and U took longer than the three
standard methods together on 14 of 16 datasets (Letter: 87 s against 9 s per seed). (reanalysis §8)

**Base-model table (§7 Base model): two new columns.** Training error at θ = 0.3 and training rows
per box (seeds 70–89): S1–S3 0.0% and 2.6; Wine 0.0% and 4.5; Breast Cancer 0.8% and 10.7; Digits
0.0% and 4.9; Thyroid 1.6% and 108; Shuttle 0.2% and 3,336; Skin 4.0% and 2,268; MAGIC 26.6% and 53;
Letter 21.3% and 28; Landsat 10.4% and 26; Page Blocks 14.4% and 176. (reanalysis §9)
