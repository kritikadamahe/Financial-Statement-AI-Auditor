import os
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_pdf(filename, company_name, industry_desc, md_a, income_data, bs_data, cf_data):
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", "pdf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    doc = SimpleDocTemplate(out_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, spaceAfter=6)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=11, textColor=colors.grey, spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=9.5, leading=13, spaceAfter=10)
    
    story = []

    # Cover
    story.append(Paragraph(company_name, title_style))
    story.append(Paragraph("Annual Report — Fiscal Years Ended December 31, 2022 through 2024", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(industry_desc, body_style))
    story.append(Spacer(1, 6))

    # MD&A
    story.append(Paragraph("Management's Discussion and Analysis", heading_style))
    for para in md_a:
        story.append(Paragraph(para, body_style))
    story.append(Spacer(1, 12))

    # Table styling function
    header_bg = colors.HexColor("#1a1a2e")
    row_bg_1 = colors.HexColor("#f0f0f5")
    row_bg_2 = colors.white
    
    def styled_table(data):
        col_w = [180, 80, 80, 80]
        t = Table(data, colWidths=col_w)
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8.5),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ]
        for i in range(1, len(data)):
            bg = row_bg_1 if i % 2 == 1 else row_bg_2
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), bg))
        t.setStyle(TableStyle(style_cmds))
        return t

    # Statements
    story.append(Paragraph("Consolidated Statements of Operations", heading_style))
    story.append(styled_table(income_data))
    story.append(Spacer(1, 20))
    story.append(PageBreak())

    story.append(Paragraph("Consolidated Balance Sheets", heading_style))
    story.append(styled_table(bs_data))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Consolidated Statements of Cash Flows", heading_style))
    story.append(styled_table(cf_data))

    doc.build(story)
    print(f"Successfully generated PDF: {out_path}")
    
    # Generate matching Ground Truth CSV
    csv_filename = filename.replace("detailed_", "").replace(".pdf", ".csv")
    if "technology" in filename: csv_filename = "technology_saas_corp.csv"
    elif "manufacturing" in filename: csv_filename = "manufacturing_industries.csv"
    elif "retail" in filename: csv_filename = "retail_ecommerce_inc.csv"
    elif "healthcare" in filename: csv_filename = "healthcare_biotech_ltd.csv"
    
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    csv_path = os.path.join(csv_dir, csv_filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(income_data[0]) # Header
        for row in income_data[1:] + bs_data[1:] + cf_data[1:]:
            writer.writerow(row)
    print(f"Successfully generated CSV: {csv_path}")


def main():
    # 1. Tech
    generate_pdf(
        "detailed_technology_saas.pdf",
        "NovaTech Solutions Inc. (Tech/SaaS)",
        "NovaTech provides enterprise cloud software and AI analytics.",
        [
            "Revenue grew rapidly due to AI product adoption. Gross margins expanded to 72% as cloud costs optimized.",
            "Operating expenses rose significantly due to R&D. Accounts receivable increased sharply in 2024 due to large enterprise deals."
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Revenue", "198,000", "252,000", "310,000"],
            ["Cost of Goods Sold", "63,360", "73,080", "86,800"],
            ["Total Expenses", "145,860", "176,080", "214,300"],
            ["Net Income", "28,080", "37,200", "43,400"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Current Assets", "100,000", "133,000", "147,000"],
            ["Total Assets", "143,000", "235,000", "255,000"],
            ["Current Liabilities", "43,500", "54,000", "65,500"],
            ["Total Debt", "9,000", "60,000", "56,000"],
            ["Total Equity", "86,500", "121,000", "133,500"],
            ["Accounts Receivable", "28,000", "35,000", "58,000"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Operating Cash Flow", "33,000", "45,000", "38,000"],
            ["Net Income", "28,080", "37,200", "43,400"]
        ]
    )

    # 2. Manufacturing
    generate_pdf(
        "detailed_manufacturing_heavy.pdf",
        "SteelForge Global (Manufacturing)",
        "SteelForge is a heavy industrial manufacturer of structural steel.",
        [
            "Revenue remained flat, but Cost of Goods Sold spiked in 2024 due to raw material inflation (iron ore tariffs).",
            "Gross margin dropped from 35% to 22%. The company took on massive debt in 2024 to survive the margin squeeze."
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Revenue", "500,000", "510,000", "495,000"],
            ["Cost of Goods Sold", "325,000", "331,500", "386,100"],
            ["Total Expenses", "120,000", "125,000", "130,000"],
            ["Net Income", "41,250", "40,125", "-15,825"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Current Assets", "150,000", "145,000", "130,000"],
            ["Total Assets", "800,000", "810,000", "850,000"],
            ["Current Liabilities", "125,000", "130,000", "160,000"],
            ["Total Debt", "300,000", "290,000", "450,000"],
            ["Total Equity", "375,000", "390,000", "240,000"],
            ["Inventory", "85,000", "90,000", "110,000"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Operating Cash Flow", "65,000", "60,000", "-25,000"],
            ["Net Income", "41,250", "40,125", "-15,825"]
        ]
    )

    # 3. Retail
    generate_pdf(
        "detailed_retail_ecommerce.pdf",
        "OmniShop Retail Group (Retail)",
        "OmniShop is a fast-fashion e-commerce retailer.",
        [
            "Aggressive marketing led to massive revenue growth, but an inventory glut in 2024 forced deep discounts.",
            "As a result, net income cratered despite high revenue. Inventory levels are alarmingly high."
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Revenue", "200,000", "350,000", "600,000"],
            ["Cost of Goods Sold", "120,000", "210,000", "450,000"],
            ["Total Expenses", "60,000", "110,000", "180,000"],
            ["Net Income", "15,000", "22,500", "-22,500"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Current Assets", "80,000", "150,000", "280,000"],
            ["Total Assets", "120,000", "200,000", "350,000"],
            ["Current Liabilities", "60,000", "120,000", "220,000"],
            ["Total Debt", "20,000", "40,000", "80,000"],
            ["Total Equity", "40,000", "40,000", "50,000"],
            ["Inventory", "40,000", "90,000", "210,000"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Operating Cash Flow", "25,000", "10,000", "-85,000"],
            ["Net Income", "15,000", "22,500", "-22,500"]
        ]
    )

    # 4. Healthcare
    generate_pdf(
        "detailed_healthcare_biotech.pdf",
        "MediLife BioSciences (Healthcare)",
        "MediLife develops advanced oncology therapeutics.",
        [
            "Revenues dropped in 2024 due to patent expiration of our flagship drug. However, R&D expenses remained high.",
            "Cash flow remains extremely strong due to licensing deals, presenting a mismatch with dropping revenues."
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Revenue", "400,000", "420,000", "250,000"],
            ["Cost of Goods Sold", "80,000", "84,000", "50,000"],
            ["Total Expenses", "200,000", "210,000", "220,000"],
            ["Net Income", "90,000", "94,500", "-15,000"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Current Assets", "250,000", "320,000", "450,000"],
            ["Total Assets", "500,000", "600,000", "700,000"],
            ["Current Liabilities", "100,000", "110,000", "90,000"],
            ["Total Debt", "50,000", "40,000", "30,000"],
            ["Total Equity", "350,000", "450,000", "580,000"],
            ["Accounts Receivable", "60,000", "65,000", "40,000"]
        ],
        [
            ["Metric", "2022", "2023", "2024"],
            ["Operating Cash Flow", "110,000", "115,000", "140,000"],
            ["Net Income", "90,000", "94,500", "-15,000"]
        ]
    )

if __name__ == "__main__":
    main()
