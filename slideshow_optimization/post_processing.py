from typing import List
from .utils import (
    calc_score,
    sequence_score,
    calc_max_score,
    lazy_calc_score,
    sequence_max_score,
)
from .models import Photo


def post_processing(data: List[Photo]):
    print("Post processing...")
    nb_attempts = 0
    previous_score = 0
    greedy = False
    while True:
        score = sequence_score(data)
        max_score = sequence_max_score(data)
        print(f"Score = {score} / {max_score}")

        if score <= previous_score:
            nb_attempts += 1
        else:
            nb_attempts = 0

        if nb_attempts >= 2:
            if not greedy:
                greedy = True
            else:
                break
        previous_score = score

        data = _improve(data[::-1], greedy=greedy)

    print("Done.")
    score = sequence_score(data)
    max_score = sequence_max_score(data)
    print(f"Score = {score} / {max_score}")

    return data


def _partial_reverse(sequence, start, end):
    return sequence[:start] + sequence[start:end][::-1] + sequence[end:]


def _do_improve(sequence, i, greedy=False):
    l1, l2 = sequence[i - 1], sequence[i]
    l12, max_l12 = lazy_calc_score(l1, l2), calc_max_score(l1, l2)
    for j in range(i + 1, len(sequence)):
        r1, r2 = sequence[j - 1], sequence[j]
        max_r12 = calc_max_score(r1, r2)
        current_max_score = max_l12 + max_r12

        max_lr1 = calc_max_score(l1, r1)
        max_lr2 = calc_max_score(l2, r2)
        new_max_score = max_lr1 + max_lr2

        if not greedy and new_max_score < current_max_score:
            continue

        r12 = lazy_calc_score(r1, r2)
        current_score = l12 + r12

        lr1 = calc_score(l1, r1)
        lr2 = calc_score(l2, r2)
        new_score = lr1 + lr2

        if new_score > current_score:
            sequence = _partial_reverse(sequence, i, j)
            break

    return sequence


def _improve(submission, greedy=False):
    p1 = submission[0]
    for i in range(1, len(submission)):
        p2 = submission[i]
        if lazy_calc_score(p1, p2) < calc_max_score(p1, p2):
            submission = _do_improve(submission, i, greedy=greedy)
        p1 = p2
    return submission
