import pandas as pd

def extract_texts(csv_file, output_txt, text_column='text'):
    df = pd.read_csv(csv_file)
    if text_column not in df:
        print(f"Column '{text_column}' not found. Columns are: {df.columns}")
        return
    df[text_column].dropna().to_csv(output_txt, index=False, header=False)
    print(f"Saved {output_txt} ({len(df)} lines)")

if __name__ == "__main__":
    extract_texts('BGL_hard_cases_normals.csv',    'BGL_hard_cases_normals.txt')
    extract_texts('BGL_hard_cases_anomalies.csv',  'BGL_hard_cases_anomalies.txt')
    extract_texts('Thunderbird_hard_cases_normals.csv',   'Thunderbird_hard_cases_normals.txt')
    extract_texts('Thunderbird_hard_cases_anomalies.csv', 'Thunderbird_hard_cases_anomalies.txt')