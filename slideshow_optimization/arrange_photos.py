import itertools
import numpy as np
from tqdm import tqdm
from typing import List
from .utils import (
    to_array,
    calc_score,
    array_score,
    sequence_score,
    lazy_calc_score,
    sequence_max_score,
    sequence_lost_score,
)
from .models import Photo, Orientation

ALL_TAGS = []


def arrange_photos(data: List[Photo]):
    global ALL_TAGS
    ALL_TAGS = sorted(set([x for x in map(lambda x: x.tags, data) for x in x]))

    photos = [x for x in data if x.orientation != Orientation.Vertical]
    vertical_photos = [x for x in data if x.orientation == Orientation.Vertical]

    print("Arranging photos...")

    np.random.seed(12)
    arranged_photos = []
    for size in sorted({len(x) // 2 * 2 for x in photos}):
        sizes = (size, size + 1)
        slide_score = size // 2
        sequence = [x for x in photos if len(x) in sizes]
        if not sequence:
            continue

        sequences = _create_sub_sequences(sequence, th=slide_score)

        nb_attempts = 0
        previous_total_score = 0
        bar = tqdm(total=len(sequences) - 1, desc=f"Processing {sizes}")
        while True:
            # subsequence post processing
            # trying to reduce number of subsequences, all subsequences must remain perfect
            nb_sequence = len(sequences)
            sequences = _stitch(sequences, th=slide_score)
            sequences = _insert(sequences, th=slide_score)
            sequences = _shuffle(sequences, th=slide_score, p=0.1)
            sequences = _partial_reverse(sequences, th=slide_score, p=0.1)
            sequences, vertical_photos = _stitch_by_vertical_photos(
                sequences,
                vertical_photos,
                th=slide_score,
                nb_proposals=20000,
                p_build=0.02,
            )

            bar.update(nb_sequence - len(sequences))

            total_score = sum(sequence_score(x) for x in sequences)
            if total_score <= previous_total_score:
                nb_attempts += 1
            else:
                nb_attempts = 0
            previous_total_score = total_score

            if len(sequences) == 1 or nb_attempts >= 50:
                break

        bar.close()

        assert all(sequence_lost_score(s) == 0 for s in sequences)

        arranged_photos += sum(sequences, [])

    print("Done.")
    print(f"Number of photos: {len(arranged_photos)}")
    score = sequence_score(arranged_photos)
    max_score = sequence_max_score(arranged_photos)
    print(f"Score = {score} / {max_score}")

    return arranged_photos, vertical_photos


def _stitch(sequences, th=1):
    """ trying to connect two different sequences """
    if len(sequences) <= 1:
        return sequences

    if th == 0:
        return [sum(sequences, [])]

    for i, j in itertools.combinations(range(len(sequences)), r=2):
        s1, s2 = sequences[i], sequences[j]

        if not s1 or not s2:
            continue

        if lazy_calc_score(s1[-1], s2[0]) >= th:
            sequences[i], sequences[j] = [], s1 + s2
            continue

        if lazy_calc_score(s1[-1], s2[-1]) >= th:
            sequences[i], sequences[j] = [], s1 + s2[::-1]
            continue

        if lazy_calc_score(s1[0], s2[0]) >= th:
            sequences[i], sequences[j] = [], s1[::-1] + s2
            continue

        if lazy_calc_score(s1[0], s2[-1]) >= th:
            sequences[i], sequences[j] = [], s1[::-1] + s2[::-1]
            continue

    return [s for s in sequences if s]


def _do_insert(s1, s2, th):
    """ trying to insert sequence 1 into sequence 2 """
    if not s1 or len(s2) <= 1:
        return False, None

    for i, p2 in enumerate(s2[1:], start=1):
        p1 = s2[i - 1]

        if lazy_calc_score(p1, s1[0]) >= th and lazy_calc_score(s1[-1], p2) >= th:
            return True, s2[:i] + s1 + s2[i:]

        if lazy_calc_score(p1, s1[-1]) >= th and lazy_calc_score(s1[0], p2) >= th:
            return True, s2[:i] + s1[::-1] + s2[i:]

    return False, None


def _insert(sequences, th):
    if len(sequences) <= 1:
        return sequences

    for i, j in itertools.product(range(len(sequences)), repeat=2):
        if i != j:
            status, combined_sequence = _do_insert(sequences[i], sequences[j], th=th)
            if status:
                sequences[i], sequences[j] = [], combined_sequence

    return [s for s in sequences if s]


def _do_partial_reverse(sequence, th=1, p=0.1):
    """ trying to reverse part of the sequence """
    if len(sequence) <= 2 or p == 0:
        return sequence

    first_photo = sequence[0]
    for i, photo in enumerate(sequence[2:], start=2):
        if calc_score(first_photo, photo) >= th:
            if np.random.random_sample() < p:
                return sequence[:i][::-1] + sequence[i:]

    return sequence


def _partial_reverse(sequences, th=1, p=0.1):
    if len(sequences) <= 1:
        return sequences

    for i in range(len(sequences)):
        sequences[i] = _do_partial_reverse(sequences[i], th=th, p=p)
        sequences[i] = _do_partial_reverse(sequences[i][::-1], th=th, p=p)

    return sequences


def _do_shuffle(s1, s2, th=1, p=0.1):
    """ trying to swap some subsequence from sequence 1 and sequence 2 """
    if not s1 or len(s2) <= 1 or p == 0:
        return s1, s2

    for i, p2 in enumerate(s2[1:], start=1):
        p1 = s2[i - 1]

        if lazy_calc_score(p1, s1[0]) >= th:
            if np.random.random_sample() < p:
                return s2[:i] + s1, s2[i:]

        if lazy_calc_score(p1, s1[-1]) >= th:
            if np.random.random_sample() < p:
                return s2[:i] + s1[::-1], s2[i:]

    return s1, s2


def _shuffle(sequences, th=1, p=0.1):
    if len(sequences) <= 1 or p == 0:
        return sequences

    for i, j in itertools.product(range(len(sequences)), repeat=2):
        if i != j:
            sequences[i], sequences[j] = _do_shuffle(
                sequences[i], sequences[j], th=th, p=p
            )

    return [s for s in sequences if s]


def _create_sub_sequences(sequence, th=1):
    """ create list of perfect subsequence """
    out = []
    if not sequence:
        return out

    sub_sequence = [sequence[0]]
    sequence = sequence[1:]
    while sequence:
        p1 = sub_sequence[-1]

        _next = None
        for i, p2 in enumerate(sequence):
            if p2 & p1 == th:
                _next = i
                break

        if _next is not None:
            p2 = sequence[_next]
            sub_sequence.append(p2)
            sequence.pop(_next)
        else:
            out.append(sub_sequence)
            sub_sequence = [sequence[0]]
            sequence = sequence[1:]

    out.append(sub_sequence)

    assert all(sequence_lost_score(s) == 0 for s in out)

    return out


def _do_stitch_by_vertical_photos(sequences, proposals, th=1, p_build=0.05):
    if len(sequences) <= 1 or len(proposals) < 1:
        return

    proposals = np.array(proposals)
    ar = to_array(proposals, ALL_TAGS)
    used_pairs = set()

    def update(_i, _j, _pair, _new_sequence):
        sequences[_i], sequences[_j] = [], _new_sequence
        used_pairs.update([x for x in _pair.id])

    def build(_i, _pair, _new_sequence):
        sequences[_i] = _new_sequence
        used_pairs.update([x for x in _pair.id])

    pair = None
    for i, j in itertools.combinations(range(len(sequences)), r=2):
        s1, s2 = sequences[i], sequences[j]
        if not s1 or not s2:
            continue

        if pair is not None:
            cond = [
                p.id[0] not in pair.id and p.id[1] not in pair.id for p in proposals
            ]
            proposals = proposals[cond]
            ar = ar[cond]

        s11 = array_score(to_array(s1[0], ALL_TAGS), ar)
        s12 = s11 if len(s1) == 1 else array_score(to_array(s1[-1], ALL_TAGS), ar)
        s21 = array_score(to_array(s2[0], ALL_TAGS), ar)
        s22 = s21 if len(s2) == 1 else array_score(to_array(s2[-1], ALL_TAGS), ar)

        cond = s12 + s21 >= th * 2
        if np.any(cond):
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            update(i, j, pair, s1 + [pair] + s2)
            continue

        cond = s12 + s22 >= th * 2
        if np.any(cond):
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            update(i, j, pair, s1 + [pair] + s2[::-1])
            continue

        cond = s11 + s21 >= th * 2
        if np.any(cond):
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            update(i, j, pair, s1[::-1] + [pair] + s2)
            continue

        cond = s11 + s22 >= th * 2
        if np.any(cond):
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            update(i, j, pair, s1[::-1] + [pair] + s2[::-1])
            continue

        cond = s11 >= th
        if np.any(cond) and np.random.random_sample() <= p_build:
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            build(i, pair, [pair] + s1)
            continue

        cond = s12 >= th
        if np.any(cond) and np.random.random_sample() <= p_build:
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            build(i, pair, s1 + [pair])
            continue

        cond = s21 >= th
        if np.any(cond) and np.random.random_sample() <= p_build:
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            build(j, pair, [pair] + s2)
            continue

        cond = s22 >= th
        if np.any(cond) and np.random.random_sample() <= p_build:
            i_pair = np.random.choice(np.where(cond)[0])
            pair = proposals[i_pair]
            build(j, pair, s2 + [pair])
            continue

        pair = None

    return used_pairs


def _stitch_by_vertical_photos(
    sequences, vertical_photos, th=1, nb_proposals=10000, p_build=0.05
):
    if len(sequences) <= 1 or not any(len(x) <= th for x in vertical_photos):
        return sequences, vertical_photos

    pair_sizes = [(x, th * 2 - x) for x in range(1, th + 1)]
    for s1, s2 in pair_sizes:
        p1 = [i for i, p in enumerate(vertical_photos) if len(p) == s1]
        p2 = [i for i, p in enumerate(vertical_photos) if len(p) == s2]
        if not p1 or not p2:
            continue

        np.random.shuffle(p1)
        np.random.shuffle(p2)

        pairs = []
        if s1 != s2:
            for i1, i2 in itertools.product(p1, p2):
                p1, p2 = vertical_photos[i1], vertical_photos[i2]
                if p1 & p2 > 0:
                    continue

                pairs.append(p1 | p2)
                if len(pairs) >= nb_proposals:
                    break
        else:
            for i1, i2 in itertools.combinations(p1, r=2):
                p1, p2 = vertical_photos[i1], vertical_photos[i2]
                if p1 & p2 > 0:
                    continue

                pairs.append(p1 | p2)
                if len(pairs) >= nb_proposals:
                    break

        used_pairs = _do_stitch_by_vertical_photos(
            sequences, pairs, th=th, p_build=p_build
        )

        # exclude used photos from vertical_photos
        vertical_photos = [x for x in vertical_photos if x.id not in used_pairs]

    assert all(sequence_lost_score(s) == 0 for s in sequences)

    return [x for x in sequences if x], vertical_photos
