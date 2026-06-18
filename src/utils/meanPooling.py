import os
import numpy as np

# Resolve BASE_PATH dynamically (3 parents up from project_thesis/src/utils/meanPooling.py to get DesigniteJava/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_BASE = os.path.join(BASE_PATH, "projects_output")
emb_raw_path = os.path.join(OUTPUT_BASE, "global_embeddings_raw.vec")
metadata_path = os.path.join(OUTPUT_BASE, "global_meta.csv")
output_final_path = os.path.join(OUTPUT_BASE, "global_embeddings_final.vec")

print("1. Parsing metadata and preparing node map...", flush=True)
node_to_methods = {}
method_sums = {}
method_counts = {}

# Read the CSV file and create an inverse mapping: Node_String -> [List of Methods it belongs to]
with open(metadata_path, "r", encoding="utf-8") as f_meta:
    for line in f_meta:
        pts = line.strip().split("|")
        if len(pts) < 6: 
            continue
            
        method_id = pts[0]
        node_ids_str = pts[5].split(",")
        
        method_sums[method_id] = None
        method_counts[method_id] = 0
        
        for n_str in node_ids_str:
            if n_str not in node_to_methods:
                node_to_methods[n_str] = []
            node_to_methods[n_str].append(method_id)

print("2. Streaming vector reading and computing Mean Pooling...", flush=True)
# Open the raw embedding file and read it line by line to keep memory footprint minimal
with open(emb_raw_path, "r", encoding="utf-8") as f_emb:
    header = f_emb.readline()
    dim = int(header.strip().split()[1])
    print(f"Detected vector dimension: {dim}", flush=True)

    count = 0
    for line in f_emb:
        p = line.strip().split()
        if len(p) > 1:
            node_str = p[0] # String ID (e.g. adempiere_30064771072)
            
            # If this node is associated with any method in the metadata...
            if node_str in node_to_methods:
                vec = np.array([float(x) for x in p[1:]])
                
                # Sum the vector coordinates directly to the method totals
                for m_id in node_to_methods[node_str]:
                    if method_sums[m_id] is None:
                        method_sums[m_id] = np.zeros(dim)
                    method_sums[m_id] += vec
                    method_counts[m_id] += 1
                
                # RAM SAVING TRICK: remove the key from the dictionary after processing it.
                # This allows memory to be progressively freed up during streaming.
                del node_to_methods[node_str]
                
        count += 1
        if count % 2000000 == 0:
            print(f"Processed {count} raw vectors...", flush=True)

print("3. Final mean computation and writing output...", flush=True)
processed_methods = 0
discarded_methods = 0

with open(output_final_path, "w", encoding="utf-8") as out_v:
    for m_id, m_sum in method_sums.items():
        if method_counts[m_id] > 0:
            mean_vec = m_sum / method_counts[m_id] # Divide the sum by the count of nodes
            out_v.write(f"{m_id} " + " ".join(map(str, mean_vec)) + "\n")
            processed_methods += 1
        else:
            discarded_methods += 1

print("\n=== POOLING PIPELINE COMPLETED ===")
print(f"File saved to: {output_final_path}")
print(f"Methods successfully saved: {processed_methods}")
print(f"Discarded methods: {discarded_methods}")
