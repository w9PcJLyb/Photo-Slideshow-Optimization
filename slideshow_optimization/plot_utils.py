from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
from .utils import calc_score, calc_max_score
from .models import Photo, Orientation


def plot(submission: List[Photo], out: str):
    total_score, total_max_score = [], []
    for p1, p2 in zip(submission[:-1], submission[1:]):
        total_score.append(calc_score(p1, p2))
        total_max_score.append(calc_max_score(p1, p2))

    plt.figure(figsize=(14, 16))

    # slide size distribution
    horizontal_hist, vertical_hist = defaultdict(int), defaultdict(int)
    for photo in submission:
        is_vertical = photo.orientation == Orientation.Combined
        hist = vertical_hist if is_vertical else horizontal_hist
        hist[len(photo)] += 1
    plt.subplot(4, 1, 1)
    plt.bar(
        horizontal_hist.keys(), horizontal_hist.values(), label="horizontal", alpha=0.5
    )
    plt.bar(vertical_hist.keys(), vertical_hist.values(), label="vertical", alpha=0.5)
    plt.xlabel("number of tags")
    plt.ylabel("number of slides")
    plt.legend()

    # score
    plt.subplot(4, 1, 2)
    plt.plot(total_score, label="score", alpha=0.5)
    plt.plot(total_max_score, label="max score", alpha=0.5)
    plt.xlabel("slide")
    plt.ylabel("score")
    plt.legend()

    # number of slides
    nb_horizontal, nb_vertical = [0], [0]
    for photo in submission:
        is_vertical = photo.orientation == Orientation.Combined
        nb_horizontal.append(nb_horizontal[-1] + (not is_vertical))
        nb_vertical.append(nb_vertical[-1] + is_vertical)
    plt.subplot(4, 1, 3)
    plt.plot(nb_horizontal, label="horizontal", alpha=0.5)
    plt.plot(nb_vertical, label="vertical", alpha=0.5)
    plt.xlabel("slide")
    plt.ylabel("number of slides")
    plt.legend()

    # loss
    horizontal_loss, vertical_loss = [0], [0]
    for score, max_score, photo in zip(total_score, total_max_score, submission):
        is_vertical = photo.orientation == Orientation.Combined
        loss = max_score - score
        horizontal_loss.append(horizontal_loss[-1] + loss * (not is_vertical))
        vertical_loss.append(vertical_loss[-1] + loss * is_vertical)
    plt.subplot(4, 1, 4)
    plt.plot(horizontal_loss, label="horizontal", alpha=0.5)
    plt.plot(vertical_loss, label="vertical", alpha=0.5)
    plt.plot(
        [sum(x) for x in zip(horizontal_loss, vertical_loss)], label="total", alpha=0.5
    )
    plt.xlabel("slide")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig(f"{out}.png")
