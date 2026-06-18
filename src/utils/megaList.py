import os
import glob

# Resolve BASE_PATH dynamically (3 parents up from project_thesis/src/utils/megaList.py to get DesigniteJava/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_BASE = os.path.join(BASE_PATH, "projects_output")
global_edgelist_path = os.path.join(OUTPUT_BASE, "global.edgelist")
global_metadata_path = os.path.join(OUTPUT_BASE, "global_meta.csv")

edgelist_files = glob.glob(os.path.join(OUTPUT_BASE, "edgelists", "*.edgelist"))

print(f"Found {len(edgelist_files)} edgelists. Starting to merge everything into two global files...", flush=True)

with open(global_edgelist_path, "w") as f_edge_out, open(global_metadata_path, "w") as f_meta_out:
    for edg_in in edgelist_files:
        filename = os.path.basename(edg_in)
        sub_id = filename.replace(".edgelist", "")
        
        met_in = os.path.join(OUTPUT_BASE, "metadata", f"{sub_id}_meta.csv")

        if not os.path.exists(met_in):
            print(f"Skipping {sub_id} because the metadata file is missing.", flush=True)
            continue
        
        with open(edg_in, "r") as f_in:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 3:
                    f_edge_out.write(f"{sub_id}_{parts[0]} {sub_id}_{parts[1]} {parts[2]}\n")

        with open(met_in, "r") as f_met_in:
            for line in f_met_in:
                pts = line.strip().split("|")
                if len(pts) < 6:
                    continue
                
                # Prefix the method ID to make it unique across projects
                pts[0] = f"{sub_id}_{pts[0]}"
                
                # Prefix the node IDs for LINE to ensure uniqueness
                old_ids = pts[5].split(",")
                new_ids = [f"{sub_id}_{nid}" for nid in old_ids if nid.strip()]
                pts[5] = ",".join(new_ids)
                
                f_meta_out.write("|".join(pts) + "\n")

        print(f"Merged {sub_id} into the mega-graph.", flush=True)

print("All done! You now have 'global.edgelist' and 'global_meta.csv'.", flush=True)
