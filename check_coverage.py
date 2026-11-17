import csv
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = csv.DictReader(io.StringIO(subprocess.check_output(["attack-to-excel", "defensive-mappings", str(path)], encoding="utf8")))
objects = {item["id"]:item for item in json.loads(path.read_text("utf8"))["objects"] if not item.get("revoked") and not item.get("x_mitre_deprecated")}
links = {(item["target_ref"],item["source_ref"]) for item in objects.values() if item["type"] == "relationship" and item["relationship_type"] == "detects"}
expected = Counter()
for tid, sid in links:
    tech, strategy = objects.get(tid,{}), objects.get(sid,{})
    if tech.get("type") != "attack-pattern" or strategy.get("type") != "x-mitre-detection-strategy":
        continue
    for aid in strategy.get("x_mitre_analytic_refs", []):
        for ref in objects.get(aid,{}).get("x_mitre_log_source_references",[]):
            expected[tid,sid,aid,ref.get("name",""),ref.get("channel",""),ref.get("x_mitre_data_component_ref","")] += 1
actual = Counter(tuple(row[key] for key in ("technique_id","detection_strategy_id","analytic_id","log_source_name","channel","data_component_id")) for row in rows)
assert actual == expected and len(actual)>0, (sum(expected.values()),sum(actual.values()))
print("Official offline ICS bundle:", sum(actual.values()),"rows identical to an independent traversal;",len({k[0] for k in actual}),"techniques;",len({k[1] for k in actual}),"strategies.")

