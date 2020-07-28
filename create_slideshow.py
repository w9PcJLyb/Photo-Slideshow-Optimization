import argparse
from slideshow_optimization import (
    utils,
    plot_utils,
    arrange_photos,
    post_processing,
    match_vertical_photos,
)


def _create_slideshow(path: str, out: str = "submission.txt", plot: bool = False):
    data = utils.read_file(path)

    slideshow, vertical_photos = arrange_photos(data)
    combine_photos = match_vertical_photos(vertical_photos)
    slideshow, _ = arrange_photos(slideshow + combine_photos)
    slideshow = post_processing(slideshow)

    score = utils.sequence_score(slideshow)
    max_score = utils.sequence_max_score(slideshow)
    print(f"# Total Score = {score} / {max_score}")

    utils.create_submission(slideshow, out)

    if plot:
        plot_utils.plot(slideshow, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input data")
    parser.add_argument("--out", default="submission.txt", help="path to output")
    parser.add_argument("--plot", action="store_true", help="display graphics")
    flags = parser.parse_args()
    print(flags)

    _create_slideshow(flags.path, flags.out, flags.plot)
