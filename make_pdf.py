import os, datetime
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# -------------------------------
# PDF класс с Unicode шрифтом
# -------------------------------
class PDFReport(FPDF):

    def __init__(self):
        super().__init__()
        # добавляем Unicode-шрифт DejaVu
        self.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', uni=True)

    def header(self):
        self.set_font("DejaVu", "B", 16)
        self.cell(0, 10, "YOLO vs FCOS Comparison Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("DejaVu", "", 10)
        self.cell(0, 5, f"Generated: {datetime.date.today()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def chapter_body(self, text):
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 6, text)
        self.ln()

# -------------------------------
# MAIN
# -------------------------------
REPORT_DIR = "artifacts/metrics"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------
# читаем CSV
yolo_csv = "artifacts/metrics/yolo_metrics_test.csv"
fcos_csv = "artifacts/metrics/fcos_metrics_test.csv"

df_yolo = pd.read_csv(yolo_csv)
df_fcos = pd.read_csv(fcos_csv)

yolo_map = df_yolo["mAP"][0]
yolo_map50 = df_yolo["mAP50"][0]
yolo_fps = df_yolo["FPS"][0]

fcos_map = df_fcos["mAP"][0]
fcos_map50 = df_fcos["mAP50"][0]
fcos_fps = df_fcos["FPS"][0]

# -------------------------------
# графики
for metric, path, title in [
    ([yolo_map, fcos_map], "map.png", "mAP comparison"),
    ([yolo_map50, fcos_map50], "map50.png", "mAP50 comparison"),
    ([yolo_fps, fcos_fps], "fps.png", "Inference speed (FPS)")
]:
    plt.figure()
    plt.bar(["YOLO", "FCOS"], metric)
    plt.title(title)
    plt.savefig(os.path.join(FIG_DIR, path))
    plt.close()

# -------------------------------
# создаём PDF
pdf = PDFReport()
pdf.add_page()

pdf.chapter_title("1. Metrics comparison")
text = f"""
YOLO:
mAP = {yolo_map}
mAP50 = {yolo_map50}
FPS = {yolo_fps}

FCOS:
mAP = {fcos_map}
mAP50 = {fcos_map50}
FPS = {fcos_fps}
"""
pdf.chapter_body(text)

# графики
pdf.add_page()
pdf.chapter_title("2. mAP comparison")
pdf.image(os.path.join(FIG_DIR, "map.png"), w=120)

pdf.add_page()
pdf.chapter_title("3. mAP50 comparison")
pdf.image(os.path.join(FIG_DIR, "map50.png"), w=120)

pdf.add_page()
pdf.chapter_title("4. FPS comparison")
pdf.image(os.path.join(FIG_DIR, "fps.png"), w=120)

# выводы
pdf.add_page()
pdf.chapter_title("5. Conclusions")

conclusion = f"""
For FCOS:
The average precision across all objects is 31.4%, with mAP50 at 44.1%.
Inference speed is approximately 9 frames per second.

For a simple model trained for 12 epochs, the results can be considered satisfactory.
mAP for small objects is only 0.042 — very low, meaning small objects are almost not detected.
mAP for large objects is 0.44 — already quite decent.

For YOLO:
     mAP     mAP50       FPS
  0.490939  0.748965  8.086550 — YOLO achieved higher accuracy after 12 epochs.
This is unexpected, since on Windows with CPU for 1 epoch, YOLO showed much lower accuracy at a much higher speed.
On CUDA, even FCOS has higher speed — but with worse accuracy.
Overall, YOLO’s accuracy at mAP 0.5 and mAP50 0.75 indicates fairly good detection with a soft threshold of 0.5.

Considering the above, if time permits, the model could be retrained to better detect small objects and account for class imbalance.
"""
pdf.chapter_body(conclusion)

# -------------------------------
output_path = os.path.join(REPORT_DIR, "report.pdf")
pdf.output(output_path)
print("PDF saved:", output_path)