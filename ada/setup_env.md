# Ada environment (one-time)

Compute nodes (Ubuntu 22.04, no internet) have no ngspice, so:

1. **ngspice bundle**: the laptop's ngspice-36 (Ubuntu 22.04 package, same glibc 2.35 as the nodes) is copied with
   its non-libc shared libraries and `share/ngspice` to `~/projects/dvd/ngspice_bundle/`; `env.sh` sets PATH,
   LD_LIBRARY_PATH, SPICE_LIB_DIR, SPICE_SCRIPTS (codemodel lines removed from `spinit`; not needed for MOS work).
2. **Python**: wheels for `requirements.txt` downloaded on the gateway (`~/.gwpip/bin/pip download ... --python-version 3.10`),
   venv built on a node: `~/projects/dvd/.venv`.
3. Code: `ada/sync_up.sh` (laptop) -> `ada/submit_chain.sh` (gateway) -> `ada/sync_down.sh` (laptop).
