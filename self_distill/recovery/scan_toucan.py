import json, glob, collections, os, sys, time
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
t0 = time.time()
root = snapshot_download("Agent-Ark/Toucan-1.5M", repo_type="dataset", max_workers=16)
print(f"downloaded to {root} in {time.time()-t0:.0f}s", flush=True)
keys = json.load(open("keys.json")); first = {k["first_user"].strip(): k for k in keys}
hits = []; subsets = collections.Counter(); servers = collections.Counter(); scanned = 0
for f in sorted(glob.glob(root + "/*/train-*.parquet")):
    cfg = f.split("/")[-2]
    t = pq.read_table(f, columns=["uuid", "subset_name", "question", "metadata"]); scanned += t.num_rows
    for r in t.to_pylist():
        subsets[(cfg, r["subset_name"])] += 1
        q = (r["question"] or "").strip()
        if q in first:
            md = r["metadata"]; md = json.loads(md) if isinstance(md, str) else (md or {})
            names = [s.get("server_name") for s in (md.get("mcp_servers") or [])]
            hits.append({"cfg": cfg, "file": os.path.basename(f), "uuid": r["uuid"], "subset": r["subset_name"], "servers": names, "traj_idx": first[q]["traj_idx"], "platform": first[q]["platform"], "n_user_ours": first[q]["n_user"]})
            for n in names: servers[n] += 1
    print(f"scanned {scanned} rows, hits {len(hits)}  ({f.split('/')[-2]}/{os.path.basename(f)})", flush=True)
json.dump(hits, open("hits.json", "w"), ensure_ascii=False)
print("\n=== subsets (config, subset_name): rows ===")
for k, v in sorted(subsets.items()): print(f"  {k}: {v}")
print("\n=== hits by (cfg, subset) ===", collections.Counter((h["cfg"], h["subset"]) for h in hits))
print("=== hits by platform ===", collections.Counter(h["platform"] for h in hits), "| distinct first messages matched:", len({h["traj_idx"] for h in hits}), "of", len(first))
print("=== server names in hits ===", servers.most_common(10))
print(f"done in {time.time()-t0:.0f}s", flush=True)
