import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from .utils import to_array
from .models import Photo, Orientation


def match_vertical_photos(photos: List[Photo], max_tags_in_photo=22):
    if not all([x.orientation == Orientation.Vertical for x in photos]):
        raise ValueError("All photos must be vertical.")

    if len(photos) % 2 > 0:
        raise ValueError("Number of photos must be odd.")

    print("Matching vertical photos...")

    np.random.seed(17)
    photos = sorted(photos, key=lambda x: -len(x))
    all_tags = sorted(set([x for x in map(lambda x: x.tags, photos) for x in x]))
    data = to_array(photos, all_tags)
    df = pd.DataFrame(data, index=photos)

    pairs = []
    bar = tqdm(total=len(df))
    while len(df) > 0:
        photo, proposals = df.iloc[0].values, df.iloc[1:].values

        num_tags_if_paired = np.sum(np.logical_or(photo, proposals), axis=1)
        overlap = np.sum(np.logical_and(photo, proposals), axis=1)

        score = (
            2 * overlap
            + 3 * (num_tags_if_paired % 2)
            + 4 * (num_tags_if_paired > max_tags_in_photo)
            + np.isin(num_tags_if_paired, (12, 13, 14, 15, 16, 17, 18, 19))
        )
        if len(score) > 1:
            best_proposals = np.where(score == np.min(score))[0]
        else:
            best_proposals = [0]

        i1, i2 = 0, 1 + np.random.choice(best_proposals)
        p1, p2 = df.index[i1], df.index[i2]
        pairs.append(p1 | p2)
        df = df.iloc[[x for x in range(len(df)) if x not in (i1, i2)]]
        bar.update(n=2)

    bar.close()

    print("Done.")

    return pairs
