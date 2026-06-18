import pandas as pd
import json
import argparse
import os
from pathlib import Path

def analyze_with_live_tracking(designite_csv, ref_miner_json, output_matches_csv, output_tracking_csv):
    if not os.path.exists(designite_csv):
        print(f"Error: Designite file not found.")
        return

    project_name = Path(designite_csv).stem
    
    if os.path.isdir(output_matches_csv):
        out_m = os.path.join(output_matches_csv, f"matches_{project_name}.csv")
    else:
        out_m = output_matches_csv if output_matches_csv.endswith('.csv') else output_matches_csv + ".csv"

    if os.path.isdir(output_tracking_csv):
        out_t = os.path.join(output_tracking_csv, f"movements_{project_name}.csv")
    else:
        out_t = output_tracking_csv if output_tracking_csv.endswith('.csv') else output_tracking_csv + ".csv"

    Path(out_m).parent.mkdir(parents=True, exist_ok=True)
    Path(out_t).parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading data from Designite ---")
    df_smells = pd.read_csv(designite_csv)
    df_smells.columns = df_smells.columns.str.strip()

    col_map = {}
    for c in ['File', 'File Path', 'Type Name', 'Class Name', 'Type', 'Class']:
        if c in df_smells.columns: col_map['file_path'] = c; break
    for c in ['Method Name', 'Method', 'Member']:
        if c in df_smells.columns: col_map['method'] = c; break
    for c in ['Code Smell', 'Smell', 'Implementation Smell', 'Design Smell']:
        if c in df_smells.columns: col_map['smell'] = c; break
    for c in ['Line', 'Line no', 'Line No', 'Start Line']:
        if c in df_smells.columns: col_map['line'] = c; break

    active_smells_map = {}

    for _, row in df_smells.iterrows():
        raw_path = str(row.get(col_map.get('file_path', ''), '')).replace('\\', '/')
        if not raw_path or raw_path == 'nan': continue
        file_name = raw_path.split('/')[-1]
        
        m_name = str(row.get(col_map.get('method'), 'N/A'))
        s_name = str(row.get(col_map.get('smell'), 'N/A'))
        
        raw_line = row.get(col_map.get('line'))
        m_line = int(raw_line) if pd.notna(raw_line) and str(raw_line).strip() else -1

        if file_name not in active_smells_map:
            active_smells_map[file_name] = []

        target_entry = next((e for e in active_smells_map[file_name] if e['full_path'] == raw_path), None)
        
        if not target_entry:
            target_entry = {"full_path": raw_path, "methods": []}
            active_smells_map[file_name].append(target_entry)
            
        method_entry = next((m for m in target_entry["methods"] if m['name'] == m_name and m['line'] == m_line), None)
        
        if not method_entry:
            method_entry = {"name": m_name, "line": m_line, "smells": set()}
            target_entry["methods"].append(method_entry)
            
        method_entry["smells"].add(s_name)

    try:
        with open(ref_miner_json, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    matches = []
    tracking_history = []
    
    INTERESTING = [
        "Change Variable Type",
        "Change Parameter Type",
        "Change Return Type",
        "Extract Method",
        "Move Method",
        "Rename Method",
        "Rename Variable",
        "Rename Parameter",
        "Extract Variable",
        "Add Parameter"
    ]

    commits = history.get('commits', [])
    # Commits are typically sorted from newest to oldest in RefactoringMiner output.
    # Reversing them allows scanning forward chronologically (oldest to newest),
    # which is necessary to dynamically track class renames/moves and update our path mapping.
    commits.reverse()

    print(f"--- Analyzing {len(active_smells_map)} smelly files and {len(commits)} commits ---")

    for i, commit in enumerate(commits, 1):
        sha = commit.get('commitId', commit.get('sha1', 'N/A'))
        
        print(f"\rLoading commit {i}/{len(commits)} (SHA: {sha[:7]})...", end="", flush=True)
        
        for ref in commit.get('refactorings', []):
            ref_type = ref['type']
            
            if ref_type in ["Rename Class", "Move Class"]:
                left_loc = ref.get('leftSideLocations', [{}])[0]
                right_loc = ref.get('rightSideLocations', [{}])[0]
                old_path = left_loc.get('filePath', '')
                new_path = right_loc.get('filePath', '')
                
                if old_path and new_path:
                    old_file_name = old_path.split('/')[-1]
                    new_file_name = new_path.split('/')[-1]

                    if old_file_name in active_smells_map:
                        for idx, entry in enumerate(active_smells_map[old_file_name]):
                            if old_path in entry['full_path']:
                                removed_entry = active_smells_map[old_file_name].pop(idx)
                                removed_entry['full_path'] = removed_entry['full_path'].replace(old_path, new_path)
                                
                                if new_file_name not in active_smells_map:
                                    active_smells_map[new_file_name] = []
                                active_smells_map[new_file_name].append(removed_entry)
                                
                                tracking_history.append({
                                    'commit': sha, 'type': ref_type,
                                    'old_path': old_path, 'new_path': new_path
                                })
                                break

            elif ref_type in INTERESTING:
                for loc in ref.get('leftSideLocations', []):
                    
                    element_type = loc.get('codeElementType', '')
                    loc_desc = loc.get('description', '').lower()
                    if 'invocation' in loc_desc or 'reference' in loc_desc or element_type == 'METHOD_INVOCATION':
                        continue

                    ref_path = loc.get('filePath', '')
                    file_name = ref_path.split('/')[-1]
                    ref_element = loc.get('codeElement')
                    
                    if not ref_element:
                        continue
                        
                    ref_start = int(loc.get('startLine', -1))
                    ref_end = int(loc.get('endLine', -1))
                    
                    if file_name in active_smells_map:
                        for entry in active_smells_map[file_name]:
                            if ref_path in entry['full_path']:
                                parts = ref_element.split('(')[0].split()
                                if not parts:
                                    continue
                                ref_base = parts[-1].strip()
                                
                                candidate_methods = [m for m in entry['methods'] 
                                                     if m['name'] != 'N/A' and m['name'].split('(')[0].strip() == ref_base]

                                if not candidate_methods:
                                    continue
                                
                                matched_method = None
                                
                                for cand in candidate_methods:
                                    if ref_start <= cand['line'] <= ref_end:    
                                        matched_method = cand
                                        break
                                
                                if not matched_method:
                                    ref_center = (ref_start + ref_end) / 2
                                    matched_method = min(candidate_methods, key=lambda c: abs(c['line'] - ref_center))

                                if matched_method:
                                    for s_name in matched_method['smells']:
                                        matches.append({
                                            'commit_sha': sha,
                                            'class': file_name.replace('.java', ''),
                                            'method': matched_method['name'],
                                            'refactoring': ref_type,
                                            'smell': s_name,
                                            'desc': ref.get('description', ''),
                                            'File': entry['full_path'],
                                            'Line no': matched_method['line']
                                        })
                                break

    cols_m = ['commit_sha', 'class', 'method', 'refactoring', 'smell', 'desc', 'File', 'Line no']
    df_matches = pd.DataFrame(matches, columns=cols_m)
    
    df_matches = df_matches.drop_duplicates(subset=['class', 'method', 'refactoring', 'smell', 'desc'])
    
    df_tracking = pd.DataFrame(tracking_history, columns=['commit', 'type', 'old_path', 'new_path']).drop_duplicates()

    df_matches.to_csv(out_m, index=False)
    df_tracking.to_csv(out_t, index=False)
    
    print("\n" + "="*40)
    print("ANALYSIS COMPLETED")
    print(f"Matches Found: {len(df_matches)}")
    print(f"Tracked movements: {len(df_tracking)}")
    print(f"Saved in: {Path(out_m).parent}")
    print("-"*40)
    print("Refactoring Details:")
    if not df_matches.empty:
        final_counts = df_matches['refactoring'].value_counts()
        for ref, count in final_counts.items():
            print(f" - {ref}: {count}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--designite', required=True)
    parser.add_argument('--refminer', required=True)
    parser.add_argument('--out_matches', default='results/')
    parser.add_argument('--out_track', default='results/')
    args = parser.parse_args()

    # Four parents up from project_thesis/src/core/miner.py to get DesigniteJava/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    analyze_with_live_tracking(str(resolve(args.designite)), str(resolve(args.refminer)), 
                               str(resolve(args.out_matches)), str(resolve(args.out_track)))
