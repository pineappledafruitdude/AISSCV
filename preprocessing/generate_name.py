import argparse
from datetime import datetime
from util import add_pipe_args

MAP = {
    "n": "--name",
    "c": "--color=",
    "f": "--folds=",
    "batch_size": "--batch_size=",
    "nbr_augment": "--augmentations=",
    "t": "--transform=",
    "occl": "--occlude=",
    "incl_no_label": "--incl_no_label=",
    "is_final": "--is_final"
}
DEFAULT = {
    "c": False,
    "f": 1,
    "batch_size": 3000,
    "nbr_augment": 10,
    "t": 1,
    "occl": False,
    "incl_no_label": False,
    "is_final": False,
}


def main(args: argparse.Namespace) -> str:
    name = datetime.today().strftime("J%Y%m%d_%H%M%S")
    output = []
    for key, value in args.__dict__.items():
        if not key in MAP or (key == 'n' and value == None):
            continue
        if key in DEFAULT and value != DEFAULT[key]:
            name += "_%s%s" % (MAP[key].replace("--",
                                                "").replace("=", ""), "_"+str(value) if not isinstance(value, bool)else "")
            output.append(
                "%s%s" % (MAP[key] if not isinstance(value, bool)else MAP[key].replace("=", ""), value if not isinstance(value, bool)else ""))

    print("\n\n--Name of the run--\n")
    print(name)
    print("\n--Arguments of the run--\n")
    for l in output:
        print(l)
    print("\n\n")
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generating Name for Training')
    add_pipe_args(parser, name_required=False)

    args = parser.parse_args()

    main(args)
