#!/usr/bin/env python
"""Run all HyperCloning validation suites.

This thin wrapper is equivalent to:

    python -m hc_validate.run_all ...
"""

from hc_validate.run_all import main


if __name__ == "__main__":
    raise SystemExit(main())
