import re

_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


def parse(tag):
    m = _SEMVER.match(tag.strip())
    if not m:
        raise ValueError(f"not semver: {tag!r}")
    major, minor, patch, pre = m.groups()
    return int(major), int(minor), int(patch), pre or ""


def bump(tag, part="patch"):
    major, minor, patch, _ = parse(tag)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
