# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Darcy Joseph
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""zeos-torch: fetch the weights the torch backend runs on."""

from __future__ import annotations

import argparse

from zeos_torch import model as model_mod


def _cmd_fetch(args: argparse.Namespace) -> int:
    path = model_mod.fetch(args.repo)
    print(f"model at {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zeos-torch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-model", help="download the Hugging Face weights")
    p_fetch.add_argument("--repo", default=model_mod.DEFAULT_REPO)
    p_fetch.set_defaults(func=_cmd_fetch)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
