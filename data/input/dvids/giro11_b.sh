#!/bin/bash
# Giro 11, dopo la revisione dell'operatore: escludi -> frame -> catena -> preselezione -> conteggi -> task CVAT.
# Uso: bash data/input/dvids/giro11_b.sh fogli_trattiN data/input/dvids/0_raw/scartiN.txt
set -e
cd "$(dirname "$0")/../../.."
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
P=".venv/bin/python tools/extra_sources.py"
$P --fonte dvids --step escludi --fogli "$1" --scarti "$2"
$P --fonte dvids --step frame --classe airplane --grandezza 96 > data/input/dvids/frame11.log 2>&1
$P --fonte dvids --step catena > data/input/dvids/catena11b.log 2>&1
$P --fonte dvids --step sessioni
.venv/bin/python tools/preselection.py
for f in dvids antiuav fbd; do $P --fonte $f --step conta > /dev/null; done
.venv/bin/python tools/fred.py --step conta > /dev/null
# task CVAT di DVIDS, Anti-UAV e FBD: si ricreano solo se nessuno e' stato ancora toccato
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, ".")
from tools.common import cvat_client
with cvat_client() as c:
    sel = [t for t in c.tasks.list() if t.name.split("_")[0] in ("fbd", "dvids", "auav")]
    touched = [t.name for t in sel if {j.state for j in t.get_jobs()} != {"new"}]
    if touched:
        sys.exit(f"task gia' toccati, non cancello: {touched[:5]}")
    for t in sel:
        t.remove()
    print("task cancellati", len(sel))
EOF
for f in dvids antiuav fbd; do $P --fonte $f --step cvat > data/input/dvids/cvat_$f.log 2>&1; grep -c '^cvat: task' data/input/dvids/cvat_$f.log; done
