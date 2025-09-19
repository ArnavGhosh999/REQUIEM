from fpdf import FPDF
import numpy as np

def save_cv_report_pdf(dataset_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, f"{dataset_name} Cross-Validation Report", ln=True)
    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 10, "CV Score Distribution (Box + Violin):", ln=True)
    pdf.image(f"{dataset_name}_cv_boxplot.png", w=140)
    pdf.cell(0, 10, "", ln=True)
    pdf.image(f"{dataset_name}_cv_violin.png", w=140)
    pdf.cell(0,10,"",ln=True)

    # Fold-wise table
    pdf.cell(0, 10, "Fold-wise F1 / AUC:", ln=True)
    try:
        f1s = np.loadtxt(f"{dataset_name}_cv_f1s.txt")
        aucs = np.loadtxt(f"{dataset_name}_cv_aucs.txt")
        pdf.set_font("Arial", "B", 10)
        pdf.cell(30,8,"Fold",0,0)
        pdf.cell(40,8,"F1 (Anomaly)",0,0)
        pdf.cell(40,8,"AUC",0,1)
        pdf.set_font("Arial", "", 10)
        for i in range(len(f1s)):
            pdf.cell(30,8,str(i+1),0,0)
            pdf.cell(40,8,f"{f1s[i]:.4f}",0,0)
            pdf.cell(40,8,f"{aucs[i]:.4f}",0,1)
        pdf.cell(30,8,"Mean",0,0)
        pdf.cell(40,8,f"{np.mean(f1s):.4f}",0,0)
        pdf.cell(40,8,f"{np.mean(aucs):.4f}",0,1)
        pdf.cell(30,8,"StdDev",0,0)
        pdf.cell(40,8,f"{np.std(f1s):.4f}",0,0)
        pdf.cell(40,8,f"{np.std(aucs):.4f}",0,1)
    except Exception as e:
        pdf.cell(0,8,f"(Could not load fold stats: {e})",ln=True)

    pdf.cell(0, 10, "Out-of-Fold Anomaly Probabilities:", ln=True)
    pdf.image(f"{dataset_name}_cv_oof_probs.png", w=140)
    pdf.cell(0,10,"",ln=True)

    pdf.output(f"{dataset_name}_cv_report.pdf")
    print(f"Saved PDF: {dataset_name}_cv_report.pdf")

# Call for each dataset
save_cv_report_pdf("BGL")
save_cv_report_pdf("Thunderbird")
