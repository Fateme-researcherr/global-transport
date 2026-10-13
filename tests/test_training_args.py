"""Tests for train_dmf.py / train_fm.py: argparser coverage (checked on the source, so no
CUDA/flash-attn needed)."""
import ast


def _argparse_dests(path):
    src = open(path).read()
    tree = ast.parse(src)
    dests = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flag = node.args[0].value
            if isinstance(flag, str) and flag.startswith("--"):
                dests.add(flag.lstrip("-").replace("-", "_"))
    return dests, src


def _args_attrs_used(src):
    used = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "args"):
            used.add(node.attr)
    return used


# --- argparser coverage: every args.X used must be defined ---------------

def test_train_dmf_argparser_defines_all_used_attrs():
    dests, src = _argparse_dests("train_dmf.py")
    used = _args_attrs_used(src)
    missing = sorted(used - dests)
    # Some attrs are set programmatically (e.g. via setattr or accelerate); keep
    # the assertion focused on the confirmed-dead/missing flags.
    # Here we only assert there is no *used-but-undefined* attr.
    # `model` is a module-level name, not args.model — filter false positives
    # by checking against the actual argparser dests.
    real_missing = [m for m in missing if m in used]
    # Allow a small whitelist of dynamically-set attrs.
    whitelist = set()
    real_missing = [m for m in real_missing if m not in whitelist]
    assert not real_missing, f"train_dmf.py uses undefined args attrs: {real_missing}"


def test_train_fm_argparser_defines_all_used_attrs():
    dests, src = _argparse_dests("train_fm.py")
    used = _args_attrs_used(src)
    missing = sorted(used - dests)
    whitelist = set()
    real_missing = [m for m in missing if m not in whitelist]
    assert not real_missing, f"train_fm.py uses undefined args attrs: {real_missing}"
