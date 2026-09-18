from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path("IBM_Case_Study_AI_vs_Real_Image_Detector.docx")
BLUE = "1F4E79"
LIGHT = "DCE6F1"


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), color)
    tc_pr.append(element)


def set_cell_margin(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value)); node.set(qn("w:type"), "dxa")


def set_widths(table, widths):
    table.autofit = False
    table.allow_autofit = False
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
            set_cell_margin(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_text(cell, text, bold=False, color=None, size=10.5):
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = bold; run.font.size = Pt(size); run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    if color: run.font.color.rgb = RGBColor.from_string(color)


def add_heading(doc, text):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.keep_with_next = True
    p.add_run(text)


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.1
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(3)


def header_footer(section):
    h = section.header.paragraphs[0]
    h.text = "IBM CASE STUDY  |  AI vs Real Image Detector"
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in h.runs:
        r.font.size = Pt(8); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(100, 100, 100)
    f = section.footer.paragraphs[0]
    f.text = "IBM Internship Case Study"
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in f.runs:
        r.font.size = Pt(8); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(100, 100, 100)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_widths(table, widths)
    for c, text in zip(table.rows[0].cells, headers):
        shade(c, BLUE); add_text(c, text, bold=True, color="FFFFFF")
    for row in rows:
        cells = table.add_row().cells
        for c, text in zip(cells, row): add_text(c, str(text))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(0.85)
    header_footer(sec)
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"; styles["Normal"].font.size = Pt(12)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    styles["Heading 1"].font.name = "Times New Roman"; styles["Heading 1"].font.size = Pt(15); styles["Heading 1"].font.color.rgb = RGBColor.from_string(BLUE)
    styles["Heading 1"]._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    styles["Heading 1"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")

    # Cover
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(105)
    r = p.add_run("IBM CASE STUDY"); r.bold = True; r.font.size = Pt(28); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor.from_string(BLUE)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("AI vs Real Image Detector"); r.italic = True; r.font.size = Pt(17); r.font.name = "Times New Roman"
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(34)
    r = p.add_run("IBM Internship - Data Science Module"); r.font.size = Pt(12); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor(80, 80, 80)
    cover = doc.add_table(rows=0, cols=2); cover.style = "Table Grid"; set_widths(cover, [2.0, 4.3])
    for a, b in [("Name", "B. Harshitha"), ("Roll No", "A23126552010"), ("IBM ID", "IBMQ2DST1697"), ("Department", "CSE (AI & ML)"), ("College Name", "Anil Neerukonda Institute of Technology and Sciences"), ("Module Name", "Data Science"), ("Module Number", "6"), ("UG Level", "U3")]:
        cells = cover.add_row().cells; shade(cells[0], LIGHT); add_text(cells[0], a, bold=True); add_text(cells[1], b)
    # The table of contents has its own intentionally centred page.
    doc.add_page_break()

    # TOC
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(155); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = p.add_run("TABLE OF CONTENTS"); r.bold = True; r.font.size = Pt(18); r.font.name = "Times New Roman"; r.font.color.rgb = RGBColor.from_string(BLUE)
    toc_rows = [("1", "Case Study Opted", "3"), ("2", "Overview", "3"), ("3", "Problem Statement", "3"), ("4", "Dataset and Methodology", "4"), ("5", "KPIs Defined", "4"), ("6", "Prototype Design", "4"), ("7", "Testing and Key Insights", "5"), ("8", "Business Impact and Recommendations", "5"), ("9", "Challenges and Solutions", "5"), ("10", "Outcome of Case Study", "6"), ("11", "Conclusion", "6")]
    add_table(doc, ["S.No", "Topic", "Page No."], toc_rows, [0.7, 4.9, 0.7])
    doc.add_page_break()

    add_heading(doc, "1. Case Study Opted")
    add_body(doc, "Case study title: AI vs Real Image Detector. This case study documents a desktop machine-learning prototype that assists a user in distinguishing ordinary photographs from AI-generated images. The tool accepts an image, extracts visual statistics, and returns a predicted class with a model-confidence score.")
    bullets(doc, ["Tools used: Python, Tkinter, OpenCV, Pandas, scikit-learn and Pillow.", "Dataset: locally organised images in dataset/real and dataset/ai.", "Deliverable: a runnable desktop prototype, training pipeline, evaluation script and model documentation."])
    add_heading(doc, "2. Overview")
    add_body(doc, "Synthetic-image tools are increasingly accessible, making it harder for students, content reviewers and ordinary users to judge whether an image is a camera photograph or computer-generated. A quick, explainable first-pass classifier can support awareness and triage, while recognising that it cannot replace forensic verification.")
    add_heading(doc, "3. Problem Statement")
    add_body(doc, "Manual visual inspection is subjective and inconsistent. The project addresses the need for a simple local application that provides a repeatable first indication from an uploaded image.")
    bullets(doc, ["No consistent, repeatable assessment when images are reviewed manually.", "AI generators, compression and editing can change visible visual patterns.", "An unbalanced or leaked dataset can make a model appear accurate when it is not reliable."])

    add_heading(doc, "4. Dataset and Methodology")
    add_body(doc, "The dataset is organised into class folders. The source-image folders contain 6,011 real images and 6,011 AI-generated images (11 in dataset/ai and 6,000 in dataset/fake). Matching processed-image folders contain 21,300 images per class. A random sample of 100 source real images and 100 source AI images was successfully opened during the input-readability check.")
    add_table(doc, ["Class", "Source images", "Processed images", "Purpose"], [("Real", "6,011", "21,300", "Photographic reference class"), ("AI-generated", "6,011", "21,300", "Synthetic reference class")], [1.3, 1.25, 1.45, 2.3])
    add_body(doc, "The feature pipeline resizes every image to 256 x 256 pixels and computes 28 interpretable features: brightness, noise, Laplacian sharpness, RGB means and standard deviations, saturation, contrast, edge density and a 16-bin grayscale histogram. Two candidates are compared: a class-balanced Random Forest and a scaled RBF Support Vector Machine.")
    add_body(doc, "To avoid data leakage, original source images are separated into stratified training and holdout sets before any augmentation. Only training images are flipped, rotated or brightness-adjusted. The final test set remains unaugmented and contains source images not used in training.")
    add_heading(doc, "5. KPIs Defined")
    add_table(doc, ["KPI", "Definition", "Why it matters"], [("Balanced accuracy", "Mean recall across both classes", "Prevents dominant real-image class from masking poor AI detection"), ("Confusion matrix", "Actual vs predicted class counts", "Shows false-positive and false-negative patterns"), ("Input readability", "Images decoded without error", "Confirms the prototype can process supported files"), ("Prediction response", "Class and confidence returned", "Measures basic user-facing prototype behaviour")], [1.45, 2.35, 2.5])
    add_heading(doc, "6. Prototype Design")
    add_body(doc, "The solution has four layers: image upload, feature extraction, model inference and result display. Tkinter provides a lightweight graphical interface where the user selects a JPG, JPEG or PNG image, previews it and presses Predict. The application then displays REAL IMAGE or AI GENERATED IMAGE along with the maximum class probability.")
    add_table(doc, ["Component", "Role"], [("app.py", "Desktop UI and upload/predict workflow"), ("feature_extraction.py", "Shared, deterministic 28-feature extraction"), ("train_model.py", "Leakage-aware training, model selection and model-card output"), ("evaluate_model.py", "Holdout-set evaluation with balanced accuracy"), ("predict.py", "Command-line prediction for one file")], [2.0, 4.3])
    add_heading(doc, "7. Testing and Key Insights")
    add_body(doc, "Testing completed in the available submission environment is summarised below. Code syntax was compiled successfully for the five Python modules. The installed runtime initially lacked OpenCV and scikit-learn; a requirements file and setup instructions were added so the complete ML run can be reproduced in a standard Python virtual environment.")
    add_table(doc, ["Test", "Result", "Evidence"], [("Dataset folder audit", "PASS", "6,011 source images per class; 21,300 processed images per class"), ("Image decode check", "PASS", "100/100 sampled source real and 100/100 sampled source AI images readable"), ("Python syntax compilation", "PASS", "app, feature extraction, train, evaluate and predict modules compiled"), ("Holdout ML metrics", "READY", "Leakage-aware training and balanced evaluation scripts are included")], [1.65, 1.2, 3.45])
    add_body(doc, "Key insight: source-level class counts are balanced when dataset/ai and dataset/fake are combined. Balanced accuracy and per-class recall remain the primary metrics because they reveal unequal class performance even when the dataset is balanced.")
    add_heading(doc, "8. Business Impact and Recommendations")
    add_body(doc, "The prototype provides a fast local screening step for educational demonstrations, media-literacy projects and preliminary review of user-submitted images. It also makes the decision process more transparent by exposing the prediction confidence rather than returning only a label.")
    bullets(doc, ["Collect at least 300 to 500 diverse AI-generated images from several generators before claiming model performance.", "Keep separate, untouched test images and report balanced accuracy, precision, recall and a confusion matrix.", "Add generator family, resolution and compression metadata to monitor performance by subgroup.", "Use the tool as an advisory signal; escalate important decisions to human or forensic review."])
    add_heading(doc, "9. Challenges and Solutions")
    add_table(doc, ["Challenge", "Solution applied"], [("Severe class imbalance", "Class-balanced models and balanced accuracy were adopted; performance claims are withheld until more AI data is collected."), ("Augmentation leakage risk", "The training workflow splits source images before augmentation."), ("Different image sizes and formats", "Images are normalised to 256 x 256 and validated during reading."), ("User-friendly inference", "A Tkinter interface and a command-line predictor were supplied.")], [2.15, 4.15])
    add_heading(doc, "10. Outcome of Case Study")
    add_body(doc, "A complete, documented AI-vs-real image detection prototype was prepared. The project now includes a desktop interface, shared feature extraction, leakage-aware model training, holdout evaluation, command-line prediction, dependency requirements, a project README and a model-card output. Dataset and source-code checks passed. The source dataset is balanced across real and AI-generated images, supporting a valid next training run and transparent reporting of holdout metrics.")
    add_heading(doc, "11. Conclusion")
    add_body(doc, "The case study demonstrates an end-to-end machine-learning workflow for AI-image screening, from data organisation and interpretable feature engineering to a usable desktop interface. The prototype is operationally structured and reproducible, but responsible reporting requires further AI-image data and an independent holdout evaluation. With those additions, the same pipeline can become a stronger comparative classifier and a useful learning tool for synthetic-media awareness.")
    add_body(doc, "References: OpenCV documentation; scikit-learn documentation; project-local dataset inventory and source code.")
    doc.core_properties.title = "IBM Case Study - AI vs Real Image Detector"
    doc.core_properties.author = "B. Harshitha"
    doc.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
