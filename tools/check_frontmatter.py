#!/usr/bin/env python3
"""Fail when a note's frontmatter does not parse as YAML.

pre-commit passes the staged notes as arguments. A note has frontmatter when its
first line is `---` and a later line closes it with `---`, the rule Obsidian uses.
A note without frontmatter passes, and so does one whose opening `---` is never
closed, since Obsidian reads that as a horizontal rule.

    tools/check_frontmatter.py notes/a.md notes/b.md
"""

import sys

import yaml


def frontmatter(text):
    """The YAML between the opening and closing `---`, or None when the note has none."""
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "---":
        return None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == "---":
            return "\n".join(lines[1:i])
    return None


def main(paths):
    broken = []
    for path in paths:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            block = frontmatter(f.read())
        if block is None:
            continue
        try:
            yaml.safe_load(block)
        except yaml.YAMLError as err:
            mark = getattr(err, "problem_mark", None)
            line = f":{mark.line + 2}" if mark else ""  # +1 for the opening `---`, +1 to count from 1
            broken.append(f"{path}{line}: frontmatter is not valid YAML ({getattr(err, 'problem', None) or err})")
    for row in broken:
        print(row)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
