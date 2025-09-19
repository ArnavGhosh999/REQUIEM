from fpdf import FPDF
import pandas as pd

def make_pdf_report(dataset_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"{dataset_name} Anomaly Detection Results", ln=True)
    pdf.set_font("Arial", "", 12)

    # Add confusion matrix & distance hist
    pdf.cell(0, 10, "Confusion Matrix:", ln=True)
    pdf.image(f"{dataset_name}_confusion_matrix.jpg", w=120)
    pdf.cell(0,10,"",ln=True)

    pdf.cell(0, 10, "Centroid Distance Histogram:", ln=True)
    pdf.image(f"{dataset_name}_distance_hist.jpg", w=120)
    pdf.cell(0,10,"",ln=True)

    # Add text results
    pdf.cell(0,10,"Classification Report/AUC:", ln=True)
    with open(f"{dataset_name}_classification_report.txt") as f:
        for line in f:
            pdf.cell(0, 8, line.strip(), ln=True)
    pdf.cell(0,10,"",ln=True)

    # Add hard cases table (optional, just the first 10)
    hard_cases = pd.read_csv(f"{dataset_name}_hard_cases_sample.csv")
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,10,"Sample Hard Cases:",ln=True)
    pdf.set_font("Arial", "", 11)
    for idx, row in hard_cases.head(10).iterrows():
        txt = f"L:{row['label']} | normal_d:{row['dist_to_normal_centroid']:.3f} | anom_d:{row['dist_to_anomaly_centroid']:.3f} | {str(row['text'])[:70]}"
        pdf.multi_cell(0,7,txt)
        pdf.cell(0,3,"",ln=True)
    
    pdf.output(f"{dataset_name}_anomaly_report.pdf")
    print(f"Saved PDF report: {dataset_name}_anomaly_report.pdf")

# Example usage
make_pdf_report("BGL")
make_pdf_report("Thunderbird")
