import os

# Resolve BASE_PATH dynamically
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
folder_path = os.path.join(BASE_PATH, "ck_merged")
# The final file will be saved here
output_file = os.path.join(folder_path, "ck_final_output.csv")

def merge_csv_files(folder, output_path):
    if not os.path.exists(folder):
        print(f"Error: The directory {folder} does not exist.")
        return

    # Retrieve all files in the folder, ignoring the output file if it already exists.
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f) != output_path]
    total_files = len(files)
    
    if total_files == 0:
        print("No files found in the specified directory.")
        return

    print(f"Starting the merge of {total_files} files...")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for index, filename in enumerate(files, start=1):
            filepath = os.path.join(folder, filename)
            
            with open(filepath, 'r', encoding='utf-8') as infile:
                # Read all lines of the current file
                lines = infile.readlines()
                
                if not lines:
                    continue # Skip completely empty files
                
                # If it is the first processed file, write everything (including the header)
                if index == 1:
                    outfile.writelines(lines)
                else:
                    # For subsequent files, write all lines except the first one (the header)
                    if len(lines) > 1:
                        outfile.writelines(lines[1:])
                
                # Check if the last read line lacks a newline and add it
                # to avoid merging the last line with the header/data of the next file
                if lines and not lines[-1].endswith('\n'):
                    outfile.write('\n')
            
            # Real-time progress tracking that bypasses buffering (flush=True)
            print(f"\rProgress: [{index}/{total_files}] Processed {filename}" + " "*10, end="", flush=True)

    print(f"\n\nMerge completed! The resulting file is located at: {output_path}")

if __name__ == "__main__":
    merge_csv_files(folder_path, output_file)
