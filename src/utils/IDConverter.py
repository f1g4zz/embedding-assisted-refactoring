import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_BASE = os.path.join(BASE_PATH, "projects_output")
global_edgelist_path = os.path.join(OUTPUT_BASE, "global.edgelist")
int_edgelist_path = os.path.join(OUTPUT_BASE, "global_int.edgelist")
mapping_path = os.path.join(OUTPUT_BASE, "node_mapping.txt")

node_map = {}
next_id = 0

print("Starting conversion of the edgelist to integer format...", flush=True)

with open(global_edgelist_path, "r") as f_in, open(int_edgelist_path, "w") as f_out:
    count = 0
    for line in f_in:
        parts = line.strip().split()
        if len(parts) >= 3:
            u, v, w = parts[0], parts[1], parts[2]
            
            # Assign an integer ID to the source node if it doesn't exist yet
            if u not in node_map:
                node_map[u] = str(next_id)
                next_id += 1
                
            # Assign an integer ID to the destination node if it doesn't exist yet
            if v not in node_map:
                node_map[v] = str(next_id)
                next_id += 1
                
            f_out.write(f"{node_map[u]} {node_map[v]} {w}\n")
            
            count += 1
            if count % 10000000 == 0:
                print(f"Processed {count} edges...", flush=True)

print("Writing mapping dictionary...", flush=True)
with open(mapping_path, "w") as f_map:
    for string_id, int_id in node_map.items():
        f_map.write(f"{int_id} {string_id}\n")

print(f"Finished! Total unique nodes: {next_id}", flush=True)
