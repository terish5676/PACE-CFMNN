import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="003366"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def create_report(output_path):
    doc = docx.Document()
    
    # Page setup - Margins: 1 inch (72 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # TITLE PAGE
    # -------------------------------------------------------------
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_top.paragraph_format.space_before = Pt(18)
    p_top.paragraph_format.space_after = Pt(4)
    run_report = p_top.add_run("A PROJECT REPORT")
    run_report.font.name = 'Times New Roman'
    run_report.font.size = Pt(16)
    run_report.font.bold = True
    run_report.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_on = doc.add_paragraph()
    p_on.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_on.paragraph_format.space_after = Pt(8)
    run_on = p_on.add_run("ON")
    run_on.font.name = 'Times New Roman'
    run_on.font.size = Pt(13)
    run_on.font.bold = True
    run_on.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(20)
    run_title = p_title.add_run(
        "PACE-CFMNN: PROVABLY AUDITABLE, CERTIFIED EXPLANATIONS AND SURGICAL SELF-REPAIR "
        "FOR CENTROID-BASED FUZZY MIN-MAX NEURAL NETWORKS"
    )
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(17)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x00, 0x33, 0x66) # Dark Navy

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_after = Pt(14)
    run_by = p_by.add_run("By")
    run_by.font.name = 'Times New Roman'
    run_by.font.size = Pt(13)
    run_by.font.italic = True

    # Student metadata table
    tbl_student = doc.add_table(rows=1, cols=2)
    tbl_student.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_student.autofit = False
    tbl_student.columns[0].width = Inches(3.2)
    tbl_student.columns[1].width = Inches(3.2)
    
    cell_s1 = tbl_student.rows[0].cells[0]
    cell_s2 = tbl_student.rows[0].cells[1]
    
    p_s1 = cell_s1.paragraphs[0]
    p_s1.paragraph_format.space_after = Pt(0)
    run_s1 = p_s1.add_run("Name of the Student:\n")
    run_s1.font.bold = True
    run_s1.font.size = Pt(11)
    run_s1_name = p_s1.add_run("I Terish Charan Tej")
    run_s1_name.font.size = Pt(12)
    run_s1_name.font.bold = True
    run_s1_name.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    p_s2 = cell_s2.paragraphs[0]
    p_s2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_s2.paragraph_format.space_after = Pt(0)
    run_s2 = p_s2.add_run("Registration no:\n")
    run_s2.font.bold = True
    run_s2.font.size = Pt(11)
    run_s2_reg = p_s2.add_run("[Registration Number]")
    run_s2_reg.font.size = Pt(12)

    # University Section
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(36)
    p_uni.paragraph_format.space_after = Pt(4)
    run_uni = p_uni.add_run("SRM UNIVERSITY, AP")
    run_uni.font.name = 'Times New Roman'
    run_uni.font.size = Pt(16)
    run_uni.font.bold = True
    run_uni.font.color.rgb = RGBColor(0x3B, 0x4B, 0x22) # SRM Olive / Gold theme

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_after = Pt(36)
    run_date = p_date.add_run("NOVEMBER 2026")
    run_date.font.name = 'Times New Roman'
    run_date.font.size = Pt(12)
    run_date.font.bold = True
    run_date.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Guidance Section
    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_after = Pt(2)
    run_g1 = p_guide.add_run("Under the guidance of\n")
    run_g1.font.italic = True
    run_g1.font.size = Pt(11)
    
    run_g2 = p_guide.add_run("Dr Vadlamudi Aadarsh\n")
    run_g2.font.bold = True
    run_g2.font.size = Pt(13)
    run_g2.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    run_g3 = p_guide.add_run("Faculty Guide, Department of Computer Science and Engineering\n"
                             "SRM University–AP, Andhra Pradesh • November 2026")
    run_g3.font.size = Pt(11)
    run_g3.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # Page Break
    doc.add_page_break()

    # -------------------------------------------------------------
    # HELPER FUNCTIONS FOR BODY HEADINGS AND CONTENT
    # -------------------------------------------------------------
    def add_h1(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
        return p

    def add_h2(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r_b = p.add_run(bold_prefix)
        r_b.font.bold = True
        r_b.font.name = 'Times New Roman'
        r_t = p.add_run(text)
        r_t.font.name = 'Times New Roman'
        return p

    # -------------------------------------------------------------
    # 1. ABSTRACT
    # -------------------------------------------------------------
    add_h1("1. Abstract")
    
    doc.add_paragraph(
        "Machine learning models are increasingly deployed in high-stakes domains such as healthcare "
        "diagnostics, automated loan underwriting, and safety-critical industrial controls. However, "
        "conventional state-of-the-art models—including Deep Neural Networks and Random Forests—operate "
        "as uninterpretable \"black boxes.\" When these models make mistakes, post-hoc explainers such as "
        "LIME and KernelSHAP produce noisy, unstable explanations that fluctuate across repeated runs. "
        "Furthermore, correcting a misclassification currently requires retraining the entire model from "
        "scratch, which is computationally expensive and risks causing \"catastrophic forgetting\" or "
        "breaking previously correct predictions."
    )
    doc.add_paragraph(
        "This project presents PACE-CFMNN (Provably Auditable, Certified Explanations for Centroid-based "
        "Fuzzy Min-Max Neural Networks), a transparent geometric machine learning framework that unifies "
        "interpretable decision-making, deterministic mathematical explanations, zero-label surgical self-repair, "
        "and formally certified safety bounds. Instead of opaque multi-layer weights, the base classifier constructs "
        "transparent, multidimensional hyper-boxes in a single online pass. When candidate hyper-boxes of opposing "
        "classes overlap, disputes are resolved using fixed empirical class centroids, completely eliminating the "
        "destructive contraction step found in historical fuzzy networks."
    )
    doc.add_paragraph(
        "The proposed approach works across four integrated stages. First, an exact Shapley value engine based on "
        "a Gray-code recurrence lattice (K · 2^d operations) achieves a 2.13x speedup and perfect rank stability "
        "(r = 1.0000), alongside PACE-flip, an analytical single-feature recourse solver that prunes 71% - 98% of "
        "boxes to deliver counterfactuals in 0.05 - 1.4 milliseconds (25x - 134x faster than standard occlusion). "
        "Second, PACE-sparse provides a combinatorial recourse algorithm that accommodates user-defined immutable "
        "feature locks (e.g., age, race) and alters 1.1 - 2.6 features on average (compared to 2.8 - 6.8 for Microsoft DiCE). "
        "Third, guided by Proposition 1 (which proves all training errors are cross-class overlap ties), the model "
        "automatically shrinks or deletes offending hyper-boxes without requiring new training labels, boosting balanced "
        "Macro-F1 by +1.5 to +6.0 points in under 4.7 seconds. Finally, every surgical edit is screened by a cached "
        "validation gate (<= 1.8 ms) and guarded by Proposition 4 spatial prediction certificates, which mathematically "
        "guarantee that outside a defined bounding box, zero existing predictions in the universe are altered. Across "
        "46,153 real-world model edits evaluated over 12,480,920 test queries, exactly zero prediction leaks occurred."
    )
    doc.add_paragraph(
        "The entire system is implemented in Python using NumPy and SciPy, leveraging native CPU SIMD vectorization "
        "(AVX-512/AVX2) with zero GPU dependencies. The model scales effortlessly to large datasets, training on 581,012 "
        "instances of Forest Covertype in 142.1 seconds (42,500 samples/second) with a peak memory footprint under 380 MB. "
        "On structured industrial benchmarks, PACE-CFMNN achieves near-parity with Random Forest (Shuttle: 99.82%, Thyroid: 97.21%) "
        "while decisively outperforming all traditional fuzzy neural baselines. This work provides an auditable, submission-ready "
        "architecture for mission-critical applications where safety, transparency, and formal verifiability are legally required."
    )
    
    p_kw = doc.add_paragraph()
    r_kw_bold = p_kw.add_run("Keywords: ")
    r_kw_bold.font.bold = True
    r_kw_bold.font.name = 'Times New Roman'
    r_kw_text = p_kw.add_run("Explainable AI (XAI), Fuzzy Min-Max Neural Networks, Shapley Values, Counterfactual Recourse, "
                             "Model Self-Repair, Formal Spatial Certification, Algorithmic Auditing.")
    r_kw_text.font.name = 'Times New Roman'

    # -------------------------------------------------------------
    # 2. INTRODUCTION
    # -------------------------------------------------------------
    add_h1("2. Introduction")
    
    doc.add_paragraph(
        "Artificial intelligence is increasingly making decisions that directly alter human lives—approving or denying "
        "mortgages, identifying cancerous tissue in radiology scans, triage-scoring ICU patients, and piloting autonomous "
        "transport. In these safety-critical environments, accuracy alone is insufficient. Regulators (such as the EU AI Act, "
        "the US Equal Credit Opportunity Act, and the FDA) mandate that machine learning systems must provide faithful, "
        "reproducible explanations and guarantee that model updates do not introduce unverified risks."
    )
    doc.add_paragraph(
        "Modern machine learning fails to satisfy these requirements for three reasons:\n"
        "1. Explanation Instability: Post-hoc explanation tools like LIME and KernelSHAP rely on random input perturbation. "
        "When queried repeatedly on the exact same patient, their feature attribution rankings fluctuate substantially "
        "(Spearman rank correlation drops to 0.24 - 0.73). In legal or medical reviews, an explanation that depends on a random "
        "seed cannot be trusted.\n"
        "2. Costly and Dangerous Model Updates: When an error is discovered in a deployed neural network or ensemble of trees, "
        "there is no localized way to patch it. Practitioners must re-run full training on all historical data. This retraining "
        "is computationally expensive and frequently causes regression bugs—silently degrading accuracy on unrelated customer groups.\n"
        "3. Absence of Spatial Safety Certificates: When a software patch is deployed in aviation or medical firmware, engineers "
        "must verify that the patch cannot harm other system modules. No such mathematical guarantee exists for deep neural networks "
        "because altering a single weight changes latent activations across the entire feature space."
    )
    doc.add_paragraph(
        "To solve these foundational problems, this project develops PACE-CFMNN. In simple terms, PACE-CFMNN represents "
        "classes as translucent, axis-aligned rectangular boxes in multi-dimensional space. If a new data point lands inside "
        "an approved box, it is classified into that category with 100% confidence. If it sits outside, confidence decays "
        "smoothly with physical distance. If boxes of opposing classes overlap, the dispute is settled by measuring Euclidean "
        "distance to empirical class centers (centroids). Because the geometry of every box is known exactly, explanations, "
        "recourse instructions, and surgical repairs can be calculated analytically without guessing."
    )

    add_h2("2.1 Objectives")
    add_bullet("Data Preparation: ", "To audit and preprocess 13 standard classification datasets, eliminating duplicate instances and applying strict, leak-free normalization into [0, 1]^d.")
    add_bullet("Interpretable Base Classifier: ", "To implement a single-pass Centroid-based Fuzzy Min-Max (CFMNN) network with fixed empirical centroids to eliminate Simpson's 1992 contraction flaws.")
    add_bullet("Exact Explanations: ", "To derive and implement exact PACE-SHAP and bounds-pruned PACE-flip, providing deterministic explanations (r = 1.0000) in sub-millisecond execution times.")
    add_bullet("Actionable Recourse: ", "To develop PACE-sparse, generating counterfactual instructions that respect locked immutable features while altering the fewest possible dials.")
    add_bullet("Proactive Error Detection: ", "To construct the PACE Refinement Score (PRS) and an error meta-classifier combining geometric boundaries with local neighborhood consensus.")
    add_bullet("Certified Surgical Self-Repair: ", "To automate zero-label model repair for cross-class overlap ties, guarded by Proposition 4 spatial prediction invariance certificates that guarantee zero regressions outside the edit zone.")

    # -------------------------------------------------------------
    # 3. METHODOLOGY
    # -------------------------------------------------------------
    add_h1("3. Methodology")
    
    doc.add_paragraph(
        "The project follows a rigorous, six-stage engineering workflow designed for transparent execution, reproducibility, "
        "and high compute efficiency:"
    )
    
    add_bullet("Stage 1 - Data Preparation & Leakage-Free Preprocessing: ", 
               "13 standardized benchmarks from UCI and OpenML were used: Iris, Wine, Glass, Breast Cancer, Ionosphere, Sonar, "
               "Vehicle, Heart Statlog, Pima Diabetes, Thyroid, Shuttle, Skin Segmentation, and Forest Covertype (581,012 rows, 54 features). "
               "An exhaustive hash-audit of the UCI Skin Segmentation dataset revealed that 193,613 out of 245,057 records (79.01%) "
               "were identical duplicate rows. These duplicates were removed to prevent artificial train-to-test leakage, establishing "
               "a clean benchmark of 51,444 unique records. All continuous features are scaled strictly into [0, 1]^d using parameters "
               "computed solely on training partitions: x_{i,k} = (x_{raw,i,k} - min X_{train}) / (max X_{train} - min X_{train} + eps).")
    
    add_bullet("Stage 2 - CFMNN Base Classifier Training: ",
               "Instances are processed sequentially in a single pass without backpropagation. A candidate hyper-box B_j = [V_j, W_j] "
               "of class y_i absorbs sample x_i if and only if max_{k=1}^d (max(W_{jk}, x_{ik}) - min(V_{jk}, x_{ik})) <= theta, "
               "where theta in [0.05, 0.60] defines maximum box dimension. If no box qualifies, a new micro-box with V_{new} = W_{new} = x_i "
               "is instantiated. Continuous proximity is evaluated via the penalty function mu_j(x) = 1.0 - (1/d) sum min(1.0, max(0.0, gamma(V_{jk}-x_k), gamma(x_k-W_{jk}))) "
               "with slope gamma = 4.0. Class scores are aggregated via max-pooling: M_c(x) = max_{j: class(B_j)=c} mu_j(x). "
               "When multiple opposing boxes tie (M_{c1} = M_{c2} = 1.0), the tie is settled by Euclidean distance to global empirical "
               "class centroids c_c = (1/|S_c|) sum x_i.")

    add_bullet("Stage 3 - Explanation & Recourse Suite (PACE): ",
               "For d <= 12, exact Shapley values are calculated via a Gray-code dynamic recurrence lattice (K · 2^d additions), "
               "achieving a 2.13x speedup over matrix methods. For d > 12, deterministic antithetic permutation pairs (32 fixed paired sequences) "
               "eliminate sampling noise. PACE-flip evaluates coordinate-independent upper bounds, pruning 71.1% - 98.2% of boxes before testing "
               "32 quantiles or solving Proposition 8 piecewise linear intersection roots in 0.05 - 1.4 ms. PACE-sparse recourse projects onto "
               "high-evidence target boxes while enforcing hard immutability locks (delta_k = 0 for locked features like age or race).")

    add_bullet("Stage 4 - Intrinsic Trust & Error Lie Detector: ",
               "Computes Prediction Disagreement (PD, 50 micro-perturbations), Explanation Instability (EI, cosine distance), "
               "Box Evidence (proximity to member training points), and Corrected Uncertainty (U*, grading ties into [0.5, 1.0]). "
               "A logistic regression meta-classifier combines these with neighborhood signals for error prediction.")

    add_bullet("Stage 5 - Surgical Self-Repair (SR): ",
               "Guided by Proposition 1 (which proves all training errors are cross-class overlap ties where mu(x in B_{true}) = mu(x in B_{false}) = 1.0), "
               "the model automatically retracts the offending box face along its dimension of minimal violation without requiring new labels.")

    add_bullet("Stage 6 - Proposition 4 Safety Certification: ",
               "Proposed edits are audited on holdout validation data (Delta Macro-F1 >= -0.5%). The engine then computes the certificate bounding envelope: "
               "C_{spatial} = prod_{k=1}^d [ min(V_{jk}, V'_{jk}) - 1/gamma, max(W_{jk}, W'_{jk}) + 1/gamma ], "
               "guaranteeing that for all x outside C_{spatial}, y_{new}(x) == y_{old}(x) provably.")

    doc.add_paragraph(
        "System Tech Stack: The framework is implemented in Python 3.11 using NumPy 1.26.4, SciPy 1.12.0, and scikit-learn 1.4.1.post1, "
        "with baselines from InterpretML (EBM) and DiCE. Computations were executed on an 8-core CPU with SIMD vectorization and zero GPU requirements. "
        "Cache-blocked chunking (B = 10,000, b = 512) was implemented for streaming 581,000+ row datasets."
    )

    # -------------------------------------------------------------
    # 4. EXPERIMENTS AND RESULTS
    # -------------------------------------------------------------
    add_h1("4. Experiments and Results")
    
    doc.add_paragraph(
        "The models were evaluated using 5-fold cross-validation and standardized train/test splits. "
        "Table 1 compares PACE-CFMNN against five major baseline paradigms across benchmark datasets. "
        "Table 2 breaks down the component-wise accuracy ablation, Table 3 details explainer efficiency and stability, "
        "and Table 4 presents the formal certification audit."
    )

    # TABLE 1: Master Accuracy Comparison
    p_t1_title = doc.add_paragraph()
    p_t1_title.paragraph_format.space_before = Pt(8)
    p_t1_title.paragraph_format.space_after = Pt(4)
    p_t1_title.paragraph_format.keep_with_next = True
    r = p_t1_title.add_run("Table 1: Master Predictive Accuracy Comparison Across Models")
    r.font.bold = True
    r.font.size = Pt(11)

    t1_data = [
        ["Benchmark Dataset", "Samples (N)", "Features (d)", "Logistic Reg (LR)", "Decision Tree (DT)", "Tuned Fuzzy (KNEFMNN)", "EBM (GA²M)", "PACE-CFMNN (Ours)", "Random Forest (200 Trees)"],
        ["Iris", "150", "4", "94.0%", "94.0%", "94.0%", "96.0%", "96.0%", "96.0%"],
        ["Wine", "178", "13", "94.4%", "88.9%", "94.4%", "97.2%", "97.2%", "97.2%"],
        ["Glass", "214", "9", "58.1%", "65.1%", "65.1%", "74.4%", "69.8%", "79.1%"],
        ["Breast Cancer", "569", "30", "94.7%", "92.1%", "93.9%", "97.4%", "95.6%", "96.5%"],
        ["Heart Statlog", "270", "13", "81.5%", "75.9%", "79.6%", "83.3%", "83.3%", "85.2%"],
        ["Pima Diabetes", "768", "8", "75.3%", "70.8%", "71.4%", "77.3%", "74.0%", "76.6%"],
        ["Thyroid", "7,200", "21", "95.2%", "96.8%", "95.6%", "97.4%", "97.2%", "97.9%"],
        ["Shuttle", "58,000", "9", "92.1%", "99.6%", "99.5%", "99.9%", "99.8%", "100.0%"],
        ["Skin (Dedup)", "51,444", "3", "91.8%", "96.5%", "94.2%", "98.1%", "94.7%", "99.9%"],
        ["Covertype", "581,012", "54", "71.2%", "78.4%", "N/A (OOM)", "82.4%", "74.1%", "95.6%"]
    ]

    table1 = doc.add_table(rows=len(t1_data), cols=len(t1_data[0]))
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table1)

    for i, row in enumerate(t1_data):
        for j, val in enumerate(row):
            cell = table1.cell(i, j)
            set_cell_margins(cell, top=80, bottom=80, left=90, right=90)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if i == 0:
                run.font.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, "003366") # Navy header
            else:
                run.font.size = Pt(9)
                if i % 2 == 1:
                    set_cell_background(cell, "F5F8FA") # Subtle stripe

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # TABLE 2: Component-wise ablation
    p_t2_title = doc.add_paragraph()
    p_t2_title.paragraph_format.space_before = Pt(8)
    p_t2_title.paragraph_format.space_after = Pt(4)
    p_t2_title.paragraph_format.keep_with_next = True
    r = p_t2_title.add_run("Table 2: Component-Wise Accuracy Progression (Ablation Analysis Across Suite)")
    r.font.bold = True
    r.font.size = Pt(11)

    t2_data = [
        ["Component Added to Pipeline", "Average Macro-F1", "Mean Gain (Δ)", "Primary Architectural Benefit"],
        ["(1) Baseline CFMNN (Naive expansion)", "80.7%", "—", "Fast single-pass online box packing."],
        ["(2) + Fixed Empirical Centroids", "83.2%", "+2.5%", "Resolves overlapping ties; eliminates presentation order bias."],
        ["(3) + Search Expansion Strategy", "84.5%", "+1.3%", "Chooses highest-membership candidate box, reducing dead space."],
        ["(4) + Surgical Self-Repair (SR)", "87.5%", "+3.0%", "Zero new labels! Surgically shrinks false-positive overlap faces."],
        ["(5) + Active Learning (AL)", "89.1%", "+1.6%", "Targeted human-in-the-loop queries on razor-thin margin ties."]
    ]

    table2 = doc.add_table(rows=len(t2_data), cols=len(t2_data[0]))
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table2)

    for i, row in enumerate(t2_data):
        for j, val in enumerate(row):
            cell = table2.cell(i, j)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (j == 1 or j == 2) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if i == 0:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, "003366")
            else:
                run.font.size = Pt(9.5)
                if i % 2 == 1:
                    set_cell_background(cell, "F5F8FA")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # TABLE 3: Explainer Benchmarks
    p_t3_title = doc.add_paragraph()
    p_t3_title.paragraph_format.space_before = Pt(8)
    p_t3_title.paragraph_format.space_after = Pt(4)
    p_t3_title.paragraph_format.keep_with_next = True
    r = p_t3_title.add_run("Table 3: Explainer Efficiency, Stability, and Recourse Benchmarking")
    r.font.bold = True
    r.font.size = Pt(11)

    t3_data = [
        ["Metric / Dimension", "Standard Baseline Tool", "PACE Implementation", "Measured Impact"],
        ["Explanation Rank Stability (r)", "LIME: 0.24 - 0.45; KernelSHAP: 0.58 - 0.73", "PACE-SHAP: 1.0000 ± 0.000", "Perfect deterministic reproducibility"],
        ["Recourse Search Latency", "Batched Occlusion: 1.3 - 162.8 ms", "PACE-flip: 0.05 - 1.4 ms", "25.2x - 134.0x speedup"],
        ["Recourse Sparsity (Features Changed)", "Microsoft DiCE: 2.8 - 6.8 features", "PACE-sparse: 1.1 - 2.6 features", "Highly digestible, actionable recourse"],
        ["Recourse Generation Queries", "Microsoft DiCE: 1,850 - 2,850 calls", "PACE-sparse: 28 - 52 calls", "100x fewer model evaluations"],
        ["Shapley Execution Operations", "Matrix MatMul: O(K · d · 2^d)", "Recurrence Lattice: O(K · 2^d)", "2.13x wall-clock speedup"],
        ["Single-Sample Inference Latency", "Random Forest: 1.2 - 14.5 ms", "PACE-CFMNN: 0.012 - 0.045 ms", "30x - 300x faster throughput"]
    ]

    table3 = doc.add_table(rows=len(t3_data), cols=len(t3_data[0]))
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table3)

    for i, row in enumerate(t3_data):
        for j, val in enumerate(row):
            cell = table3.cell(i, j)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if i == 0:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, "003366")
            else:
                run.font.size = Pt(9.5)
                if i % 2 == 1:
                    set_cell_background(cell, "F5F8FA")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # TABLE 4: Certification Audit
    p_t4_title = doc.add_paragraph()
    p_t4_title.paragraph_format.space_before = Pt(8)
    p_t4_title.paragraph_format.space_after = Pt(4)
    p_t4_title.paragraph_format.keep_with_next = True
    r = p_t4_title.add_run("Table 4: Formal Safety and Certification Verification Audit")
    r.font.bold = True
    r.font.size = Pt(11)

    t4_data = [
        ["Audit Dimension", "Evaluated Quantity", "Observed Result", "Formal Mathematical Status"],
        ["Total Surgical Edits Audited", "46,153 empirical edits", "46,153 completed", "All covered by Proposition 4 certificates"],
        ["Holdout Queries Outside Certs", "12,480,920 queries", "Zero prediction changes", "100.000% Soundness (Zero leaks)"],
        ["Average Certificate Volume Footprint", "Percentage of [0, 1]^d hypercube", "0.82% average volume", "99.18% of feature space certified untouched"],
        ["Proposition 1 Error Verification", "347,339 training errors (520 models)", "347,339 cross-class ties", "100.000% verified empirical theorem"],
        ["Condition S Monotonicity Rate", "46,153 edits", "Holds in only 1.90% of edits", "Proves empirical Validation Gate is mandatory"]
    ]

    table4 = doc.add_table(rows=len(t4_data), cols=len(t4_data[0]))
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table4)

    for i, row in enumerate(t4_data):
        for j, val in enumerate(row):
            cell = table4.cell(i, j)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.runs[0]
            run.font.name = 'Times New Roman'
            if i == 0:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_background(cell, "003366")
            else:
                run.font.size = Pt(9.5)
                if i % 2 == 1:
                    set_cell_background(cell, "F5F8FA")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Detailed discussion
    add_h2("Discussion of Experimental Findings")
    doc.add_paragraph(
        "1. Dominance in Safety & Certification: While Random Forest achieves an accuracy edge of 0.2% - 21.5% on complex, "
        "noisy non-axis-aligned manifolds, it remains a completely uncertifiable black box. PACE-CFMNN is the only architecture "
        "evaluated that allows post-deployment model editing with 100% provable spatial containment. Across 46,153 empirical edits, "
        "zero unintended prediction changes were observed outside the certified bounds.\n"
        "2. Superiority Over Classical Fuzzy Baselines: CFMNN decisively outperforms classical Simpson FMNN (+4.0% to +11.6% accuracy gain) "
        "and modern variants (EFMNN, KNEFMNN, RFMNN, IOL-GFMM) across all tested benchmarks. This confirms that eliminating Simpson's "
        "destructive contraction step in favor of empirical class centroids completely resolves historical fragmentation pathologies.\n"
        "3. High-Speed Real-Time Auditing: PACE-flip delivers exact single-feature counterfactual recourse in under 1.4 ms, representing a "
        "25x to 134x speedup over batched brute-force occlusion. This enables interactive, real-time explainability dashboards running locally "
        "on consumer laptops with zero cloud latency."
    )

    # -------------------------------------------------------------
    # 5. CONCLUSION
    # -------------------------------------------------------------
    add_h1("5. Conclusion")
    
    doc.add_paragraph(
        "This project demonstrated that machine learning can achieve high predictive performance on tabular data without "
        "sacrificing mathematical transparency, explanation stability, or post-deployment safety. By combining axis-aligned "
        "hyper-boxes with fixed centroid tie-breaking, PACE-CFMNN overcomes the decades-old instability of fuzzy min-max networks."
    )
    doc.add_paragraph(
        "The PACE framework provides the machine learning community with a complete, certified ecosystem: feature attributions "
        "are mathematically exact and perfectly stable (r = 1.0000); recourse recommendations are generated in sub-milliseconds "
        "while respecting immutable user attributes; production mistakes can be surgically self-repaired in seconds without "
        "paying for new training labels; and updates are legally and mathematically certified through Proposition 4 spatial "
        "bounding envelopes."
    )
    doc.add_paragraph(
        "Limitations: The primary limitation is the axis-aligned constraint: because hyper-boxes cannot rotate obliquely, diagonal "
        "decision boundaries require a staircase approximation, increasing box counts on correlated features. Furthermore, exact "
        "Shapley computation remains #P-hard (Proposition 5), requiring antithetic permutation sampling when dimensionality exceeds d = 12."
    )
    doc.add_paragraph(
        "Future Work: Future work will explore integrating principal component rotation matrices for oblique hyper-boxes, developing "
        "GPU-accelerated Triton kernels for multi-million-row streaming datasets, and deriving formal differential privacy bounds "
        "for hyper-box coordinates during active learning."
    )

    # -------------------------------------------------------------
    # 6. CONTRIBUTION
    # -------------------------------------------------------------
    add_h1("6. Contribution")
    
    add_bullet("Fixed Centroid Tie-Breaking Architecture: ", 
               "Eliminated Simpson's 1992 contraction mechanism, establishing a stable, order-invariant single-pass online learning "
               "framework that beats existing fuzzy baselines across 10 out of 10 benchmark datasets.")
    add_bullet("Exact & Sub-Millisecond PACE Explainers: ", 
               "Implemented the Gray-code recurrence engine (2.13x speedup) and quantile bounds pruning (25x - 134x speedup), "
               "delivering deterministic explanations (r = 1.0000) in under 1.4 milliseconds.")
    add_bullet("Constrained Actionable Recourse (PACE-sparse): ", 
               "Developed a greedy combinatorial recourse algorithm supporting immutable feature locks that reduces modified features "
               "by over 50% compared to Microsoft DiCE.")
    add_bullet("Proposition 1 Error Formulation: ", 
               "Discovered and proved that 100% of training errors are cross-class overlap ties, providing the theoretical foundation for surgical repair.")
    add_bullet("Zero-Label Surgical Self-Repair: ", 
               "Engineered an automated model refinement pipeline that boosts balanced Macro-F1 by +1.5 to +6.0 points in < 4.7 seconds "
               "without requiring manual annotation.")
    add_bullet("Formal Spatial Prediction Certificates (Proposition 4): ", 
               "Formulated and empirically proved spatial invariance certificates, verifying zero prediction leaks across 46,153 model edits "
               "and 12.4 million test evaluations.")
    add_bullet("Comprehensive 46-Volume Open Documentation Suite: ", 
               "Authored a complete, fully cross-referenced research documentation library in the repository, making the entire project "
               "100% auditable and peer-review ready.")

    # Approval block
    p_app = doc.add_paragraph()
    p_app.paragraph_format.space_before = Pt(36)
    p_app.paragraph_format.keep_with_next = True
    r_app = p_app.add_run("Report Approved by Faculty Guide:\n\n\n\n*(Signature)*\n")
    r_app.font.name = 'Times New Roman'
    r_app.font.size = Pt(11)
    
    r_app_name = p_app.add_run("Dr Vadlamudi Aadarsh\n")
    r_app_name.font.name = 'Times New Roman'
    r_app_name.font.bold = True
    r_app_name.font.size = Pt(12)
    
    r_app_dept = p_app.add_run("Department of Computer Science and Engineering\n"
                               "SRM University–AP, Andhra Pradesh")
    r_app_dept.font.name = 'Times New Roman'
    r_app_dept.font.size = Pt(11)

    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    create_report("SRM_PROJECT_REPORT_PACE_CFMNN.docx")
    create_report("research_breakdown/SRM_PROJECT_REPORT_PACE_CFMNN.docx")
    create_report("pace-cfmnn-main/research_breakdown/SRM_PROJECT_REPORT_PACE_CFMNN.docx")
