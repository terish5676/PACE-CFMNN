"""A literal, loop-by-loop transcription of the paper's pseudocode.

Written from Aadarsh & Prasad (PReMI 2023), Eq. 1, Eq. 2 and Algorithms 2
and 4, with plain Python lists and no numpy, so it shares no code with
``pace_cfmnn.cfmnn_core``. Tests check that the fast implementation builds
exactly the same boxes and makes the same predictions.
"""

import math


def membership(a, V, W, gamma=4.0):
    """Eq. 1: b_j(a) = 1/(2n) * sum_i [f(a_i - w_ji) + f(v_ji - a_i)]."""
    n = len(a)
    total = 0.0
    for i in range(n):
        total += max(0.0, 1.0 - max(0.0, gamma * min(1.0, a[i] - W[i])))
        total += max(0.0, 1.0 - max(0.0, gamma * min(1.0, V[i] - a[i])))
    return total / (2 * n)


def train(X, y, theta=0.3, gamma=4.0, fix_centroid=False, search=False):
    """Algorithm 2 for one partition. Returns boxes in creation order.

    ``search=True`` is not the paper: it tries same-class boxes in decreasing
    membership and expands the first that passes Eq. 2 (Simpson's FMNN).
    """
    boxes = []
    n = len(X[0])
    for a, dec in zip(X, y):
        a = list(a)
        # a. Highest membership among boxes of the same class (first box on ties).
        h_max, b_max = None, -1.0
        for box in boxes:
            if box["dec"] != dec:
                continue
            b = membership(a, box["V"], box["W"], gamma)
            if b > b_max:
                h_max, b_max = box, b
        # b. No box of this class yet: create a point box.
        if h_max is None:
            boxes.append({"V": a[:], "W": a[:], "dec": dec, "Sum": a[:], "Count": 1})
            continue
        if b_max == 1.0:
            if fix_centroid:  # the blueprint's fix; the paper just continues
                h_max["Sum"] = [s + ai for s, ai in zip(h_max["Sum"], a)]
                h_max["Count"] += 1
            continue
        if search:
            same = [box for box in boxes if box["dec"] == dec]
            ranked = sorted(same, key=lambda box: -membership(a, box["V"], box["W"], gamma))
            h_max = None
            for box in ranked:  # sorted() is stable: first box on ties
                size = sum(max(w, ai) - min(v, ai) for v, w, ai in zip(box["V"], box["W"], a))
                if size <= n * theta:
                    h_max = box
                    break
            if h_max is None:
                boxes.append({"V": a[:], "W": a[:], "dec": dec, "Sum": a[:], "Count": 1})
                continue
            size = 0.0
        else:
            # Eq. 2: expansion criterion on the best box only.
            size = sum(max(w, ai) - min(v, ai) for v, w, ai in zip(h_max["V"], h_max["W"], a))
        if size <= n * theta:
            h_max["V"] = [min(v, ai) for v, ai in zip(h_max["V"], a)]
            h_max["W"] = [max(w, ai) for w, ai in zip(h_max["W"], a)]
            h_max["Sum"] = [s + ai for s, ai in zip(h_max["Sum"], a)]
            h_max["Count"] += 1
        else:
            boxes.append({"V": a[:], "W": a[:], "dec": dec, "Sum": a[:], "Count": 1})
    for box in boxes:
        box["Cent"] = [s / box["Count"] for s in box["Sum"]]
    return boxes


def predict_one(x, boxes, gamma=4.0):
    """Algorithm 4: highest membership; ties go to the nearest centroid."""
    bs = [membership(list(x), box["V"], box["W"], gamma) for box in boxes]
    top = max(bs)
    win_list = [k for k, b in enumerate(bs) if b == top]
    if len(win_list) == 1:
        best = win_list[0]
    else:
        best = min(win_list, key=lambda k: math.dist(x, boxes[k]["Cent"]))
    return boxes[best]["dec"], best
