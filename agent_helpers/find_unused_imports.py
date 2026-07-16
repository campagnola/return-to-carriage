"""Rough unused-import finder: reports imported names never mentioned again in the module source."""
import ast
import sys


def check(path):
    src = open(path).read()
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.append((a.asname or a.name.split('.')[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.append((a.asname or a.name, node.lineno))

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            pass
    # attribute bases are Names already; also catch dotted usage in strings? no.
    for name, lineno in imported:
        # count occurrences in source beyond the import line itself
        uses = [i + 1 for i, line in enumerate(src.splitlines()) if name in line and (i + 1) != lineno]
        if not uses:
            print("%s:%d: '%s' appears unused" % (path, lineno, name))


for path in sys.argv[1:]:
    check(path)
