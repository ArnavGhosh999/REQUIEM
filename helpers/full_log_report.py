from fpdf import FPDF

def save_fulllog_report_pdf(dataset_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, f"{dataset_name} Full Log Classification Report", ln=True)
    pdf.set_font("Arial", "", 12)

    # Confusion Matrix
    pdf.cell(0,10,"Confusion Matrix:",ln=True)
    pdf.image(f"{dataset_name}_full_confmat.png", w=120)
    pdf.cell(0,10,"",ln=True)

    # Classification Report
    pdf.cell(0,10,"Classification Report:",ln=True)
    with open(f"{dataset_name}_full_classification_report.txt") as f:
        for line in f:
            pdf.cell(0,8,line.strip(),ln=True)
    pdf.cell(0,10,"",ln=True)

    # ROC Curve
    pdf.cell(0,10,"ROC Curve:",ln=True)
    pdf.image(f"{dataset_name}_full_roc.png", w=120)
    pdf.cell(0,10,"",ln=True)

    # Probability Histogram
    pdf.cell(0,10,"Prediction Confidence Histogram:",ln=True)
    pdf.image(f"{dataset_name}_full_conf_hist.png", w=120)
    pdf.cell(0,10,"",ln=True)

    pdf.output(f"{dataset_name}_full_report.pdf")
    print(f"Saved PDF: {dataset_name}_full_report.pdf")

# Example usage:
save_fulllog_report_pdf("BGL_full")
save_fulllog_report_pdf("Thunderbird_full")
