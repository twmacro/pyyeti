#!/usr/bin/env python

from argparse import ArgumentParser

from pyyeti.nastran import op4


def main() -> None:
    """Main."""
    parser = ArgumentParser(description="List contents of op4 file(s)")

    parser.add_argument(
        "input_file", metavar="INFILE", help="file to read", type=str, nargs="+"
    )

    args = parser.parse_args()

    for name in args.input_file:
        names, sizes, forms, mtypes = op4.dir(name, verbose=False)
        print()
        print(f"{name}: {len(names)} matrices")
        for n, s, f, m in zip(names, sizes, forms, mtypes):
            print(f"{n:8}, {s[0]:6} x {s[1]:<6}, form={f}, mtype={m}")


if __name__ == "__main__":
    main()
