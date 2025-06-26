#!/usr/bin/env python3
import argparse
import sys
import re

def process_line(line: str, output_mode: bool) -> str:
    """
    Given a single input line containing 'IN: ... OUT: ...',
    extract the appropriate segment and transform tokens if needed.
    If output_mode is False, return the IN part;
    if True, return the transformed OUT part.
    """
    m = re.match(r'.*?IN:\s*(.*?)\s*OUT:\s*(.*)', line)
    if not m:
        return ""  # skip malformed lines

    in_part, out_part = m.group(1), m.group(2)
    if not output_mode:
        return in_part
    return out_part
    # transform tokens: remove leading I_ and replace underscores with +
    # toks = out_part.split()
#     transformed = []
#     for tok in toks:
#         if tok.startswith("I_"):
#             tok = tok[2:]
#         tok = tok.replace("_", "+")
#         transformed.append(tok)
#     return " ".join(transformed)

def main():
    parser = argparse.ArgumentParser(
        description="Extract and optionally transform IN/OUT segments per line from stdin."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        action="store_true",
        help="Print the IN segment for each line"
    )
    group.add_argument(
        "--output",
        action="store_true",
        help="Print the transformed OUT segment for each line"
    )
    args = parser.parse_args()

    # Loop over each line in stdin
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        result = process_line(line, output_mode=args.output)
        if result:
            print(result)

if __name__ == "__main__":
    main()
