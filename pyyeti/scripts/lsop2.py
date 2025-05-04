from argparse import ArgumentParser

from pyyeti.nastran import op2


def main() -> None:
    """Main."""
    parser = ArgumentParser(description="List contents of op2 file(s)")

    parser.add_argument(
        "input_file", metavar="INFILE", help="file to read", type=str, nargs="+"
    )

    parser.add_argument(
        "-w",
        "--with-headers",
        dest="with_headers",
        action="store_true",
        default=False,
        help="include print of table record headers",
    )

    parser.add_argument(
        "-t",
        "--with-trailers",
        dest="with_trailers",
        action="store_true",
        default=False,
        help="include data block trailers in print out",
    )

    args = parser.parse_args()

    for name in args.input_file:
        with op2.OP2(name) as o2:
            print()
            print(f"{name}: {len(o2.dblist)} data blocks")
            o2.directory(
                with_headers=args.with_headers, with_trailers=args.with_trailers
            )


if __name__ == "__main__":
    main()
