import numpy as np
from typing import List, Callable, Union
from functools import lru_cache
from .models import Photo, Orientation


def calc_score(p1: Photo, p2: Photo) -> int:
    return min(p1 & p2, p1 - p2, p2 - p1)


@lru_cache(maxsize=2 ** 20)
def lazy_calc_score(p1: Photo, p2: Photo) -> int:
    return calc_score(p1, p2)


def calc_max_score(p1: Photo, p2: Photo) -> int:
    return min(len(p1), len(p2)) // 2


def calc_lost_score(p1: Photo, p2: Photo) -> int:
    return p1.max_score() + p2.max_score() - 2 * calc_score(p1, p2)


def _apply(sequence: List[Photo], function: Callable[[Photo, Photo], int]) -> int:
    if len(sequence) <= 1:
        return 0
    return sum(function(sequence[i], sequence[i - 1]) for i in range(1, len(sequence)))


def sequence_score(sequence: List[Photo]) -> int:
    return _apply(sequence, calc_score)


def sequence_max_score(sequence: List[Photo]) -> int:
    return _apply(sequence, calc_max_score)


def sequence_lost_score(sequence: List[Photo]) -> int:
    return _apply(sequence, calc_lost_score)


def to_array(_x: Union[List[Photo], Photo], all_tags: list) -> np.array:
    """
    Convert a photo (or list of photos) to numpy array
    all_tags -- list of all unique tags that we can meet
    """
    if isinstance(_x, Photo):
        tags = _x.tags
        array = np.zeros(len(all_tags), dtype=np.bool)
        for j, t in enumerate(all_tags):
            if t in tags:
                array[j] = True
    else:
        array = np.zeros((len(_x), len(all_tags)), dtype=np.bool)
        for i, photo in enumerate(_x):
            tags = photo.tags
            for j, t in enumerate(all_tags):
                if t in tags:
                    array[i, j] = True
    return array


def array_score(v: np.array, ar: np.array) -> np.array:
    """
    Calculate score between photo vektor (v) and photo array (ar)
    """
    return np.minimum.reduce(
        [
            np.sum(np.logical_and(v, ar), axis=1),
            np.sum(np.logical_and(v, np.logical_not(ar)), axis=1),
            np.sum(np.logical_and(np.logical_not(v), ar), axis=1),
        ]
    )


def read_file(path: str) -> List[Photo]:
    data = []
    with open(path, "r") as file:
        file.readline()  # number of photos
        for i, line in enumerate(file):
            data.append(Photo.from_string(i, line))
    return data


def check_sequence(sequence: List[Photo]):
    all_id = set()
    for photo in sequence:
        assert photo.orientation in (
            Orientation.Combined,
            Orientation.Horizontal,
        ), f"Wrong orientation: {photo}"

        photo_id = photo.id
        assert isinstance(photo_id, (int, tuple)), f"Wrong id format: {photo_id}"

        if isinstance(photo_id, tuple):
            assert len(photo.id) == 2, f"Wrong id format: {photo_id}"
            assert photo_id[0] != photo_id[1], f"Wrong id format: {photo_id}"
        else:
            photo_id = (photo_id,)

        for x in photo_id:
            assert x not in all_id, f"id {x} not unique"
            all_id.add(x)


def create_submission(submission: List[Photo], path="submission.txt"):
    check_sequence(submission)
    with open(path, "w+") as f:
        f.write("{}\n".format(len(submission)))
        for photo in submission:
            photo_id = photo.id
            if not isinstance(photo_id, tuple):
                photo_id = (photo_id,)
            f.write("{}\n".format(" ".join(map(str, photo_id))))
