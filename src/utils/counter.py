import pandas as pd
import argparse
import sys
from pathlib import Path

def main(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        
        required_columns = {'class_name', 'refactoring'}
        if not required_columns.issubset(df.columns):
            print(f"Error: file doesn't have the required columns: {required_columns}")
            sys.exit(1)

        unique_classes_count = df['class_name'].nunique()
        print(f"Total unique classes: {unique_classes_count}")

        class_analysis = df.groupby('class_name')['refactoring'].agg(
            count='count', 
            types=lambda x: list(x.unique())
        ).reset_index()

        class_analysis.columns = ['Class', 'Refactoring Count', 'Refactoring Types']
        class_analysis = class_analysis.sort_values(by='Refactoring Count', ascending=False)
        class_analysis.to_csv(output_path, index=False)
        print(f"Analysis completed. Results saved in: {output_path}")

        print("\nFirst 10 rows:")
        print(class_analysis.head(10))

    except FileNotFoundError:
        print(f"Error: file '{input_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis of refactoring by class and CSV file.")
    parser.add_argument('-i', '--input', required=True, help="path of CSV file for input")
    parser.add_argument('-o', '--output', required=True, help="path of CSV file for output")

    args = parser.parse_args()
    
    # Four parents up from project_thesis/src/utils/counter.py to get DesigniteJava/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    
    def resolve_path(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    main(resolve_path(args.input), resolve_path(args.output))
