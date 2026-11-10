"""Command-line interface.

    nuisauc list                         experiments, where they appear, what they need
    nuisauc show NAME                    description, outputs and default configuration
    nuisauc run NAME [NAME ...] | all    run experiments, write tables to --out
    nuisauc verify NAME [...] | all      compare written tables with the reference tables
    nuisauc audit                        check the manuscript's numbers against the tables
    nuisauc figures                      draw the paper's figure(s)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from nuisauc import experiments, paths
from nuisauc.io import compare_tables, read_table

log = logging.getLogger("nuisauc")


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def _config(name: str, config_file: Path | None, overrides: list[str]) -> dict:
    cfg = experiments.get(name).default_config()
    if config_file is not None:
        cfg.update(yaml.safe_load(Path(config_file).read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        if key not in cfg:
            raise SystemExit(f"{name}: unknown config key {key!r}; known: {sorted(cfg)}")
        cfg[key] = _parse_value(value)
    return cfg


def _names(args_names: list[str]) -> list[str]:
    if args_names == ["all"]:
        return experiments.available()
    for n in args_names:
        experiments.get(n)
    return args_names


def cmd_list(_args) -> int:
    rows = []
    for n in experiments.NAMES:
        try:
            m = experiments.get(n)
        except ModuleNotFoundError:
            continue
        rows.append((n, m.REQUIRES, m.PAPER, m.DESCRIPTION))
    w = max(len(r[0]) for r in rows)
    print(f"{'experiment':<{w}}  needs  {'paper':<16}  description")
    for n, req, paper, desc in rows:
        print(f"{n:<{w}}  {req:<5}  {paper:<16}  {desc}")
    return 0


def cmd_show(args) -> int:
    m = experiments.get(args.name)
    print(f"{m.NAME}: {m.DESCRIPTION}\npaper: {m.PAPER}\nneeds: {m.REQUIRES}")
    print(f"outputs: {', '.join(m.OUTPUTS)}\ndefault config:")
    print(yaml.safe_dump(m.default_config(), sort_keys=False).rstrip())
    return 0


def cmd_run(args) -> int:
    out = Path(args.out or paths.results_dir())
    for n in _names(args.names):
        m = experiments.get(n)
        cfg = _config(n, args.config, args.set or [])
        log.info("running %s -> %s", n, out)
        t0 = time.time()
        tables = m.run(cfg, out)
        log.info("%s finished in %.0f s (%s)", n, time.time() - t0, ", ".join(tables))
        if hasattr(m, "summarize"):
            print(m.summarize(tables))
    return 0


def cmd_verify(args) -> int:
    out = Path(args.out or paths.results_dir())
    ref = Path(args.reference or paths.reference_dir())
    ok = True
    report = {}
    for n in _names(args.names):
        m = experiments.get(n)
        for t in m.OUTPUTS:
            new_p, ref_p = out / f"{t}.csv", ref / f"{t}.csv"
            if not new_p.exists():
                print(f"{t}: not run (missing {new_p})")
                ok = False
                continue
            if not ref_p.exists():
                print(f"{t}: no reference table ({ref_p})")
                continue
            c = compare_tables(read_table(out, t), read_table(ref, t), t, args.atol)
            print(c)
            report[t] = {"equal": c.equal, "max_abs_diff": c.max_abs_diff,
                         "problems": c.problems}
            ok &= c.equal
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if ok else 1


def cmd_audit(args) -> int:
    from nuisauc import audit
    return audit.main(Path(args.manuscript) if args.manuscript else None,
                      Path(args.reference) if args.reference else None)


def cmd_figures(args) -> int:
    from nuisauc import figures
    for p in figures.draw_all(Path(args.out), Path(args.reference or paths.reference_dir())):
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nuisauc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiments").set_defaults(func=cmd_list)
    s = sub.add_parser("show", help="show one experiment")
    s.add_argument("name")
    s.set_defaults(func=cmd_show)
    r = sub.add_parser("run", help="run experiments")
    r.add_argument("names", nargs="+", help="experiment names or 'all'")
    r.add_argument("--out", help="output directory (default $NUISAUC_RESULTS or runs/)")
    r.add_argument("--config", type=Path, help="YAML file overriding the defaults")
    r.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override one config value (repeatable)")
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("verify", help="compare outputs with reference tables")
    v.add_argument("names", nargs="+")
    v.add_argument("--out", help="directory holding the new tables")
    v.add_argument("--reference", help="reference directory (default results/)")
    v.add_argument("--atol", type=float, default=1e-9)
    v.add_argument("--json", help="also write the comparison as JSON")
    v.set_defaults(func=cmd_verify)
    a = sub.add_parser("audit", help="check manuscript numbers against the tables")
    a.add_argument("--manuscript", help="main.tex (default manuscript/main.tex)")
    a.add_argument("--reference", help="table directory (default results/)")
    a.set_defaults(func=cmd_audit)
    f = sub.add_parser("figures", help="draw the paper's figures")
    f.add_argument("--out", default="figures")
    f.add_argument("--reference", help="table directory (default results/)")
    f.set_defaults(func=cmd_figures)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
