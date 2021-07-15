import argparse
from pathlib import Path
import os


def main(path: Path):
    jpg_images = path.glob("*.jpg")
    for img in jpg_images:
        txt = Path(path, "%s.txt" % img.stem)
        touch(txt)


def touch(path: Path):
    with open(path, 'a'):
        os.utime(path, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create empty txt files for no label images')
    parser.add_argument('-i', metavar='input folder', type=str, required=True,
                        help='Path to no label images')

    args = parser.parse_args()

    main(Path(args.i))
