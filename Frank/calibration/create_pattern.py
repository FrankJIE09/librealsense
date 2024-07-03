import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# 定义棋盘格的大小
rows = 7  # 内部行数
cols = 10  # 内部列数
square_size = 25  # 单个方格的大小（毫米）

# 创建一个黑白棋盘格图案
pattern_size = (cols, rows)
img_size = (cols * square_size, rows * square_size)
chessboard = np.zeros(img_size, dtype=np.uint8)

# 填充棋盘格
for i in range(rows+3):
    for j in range(cols):
        if (i + j) % 2 == 0:
            cv2.rectangle(chessboard, (j * square_size, i * square_size),
                          ((j + 1) * square_size, (i + 1) * square_size), 255, -1)

# 将棋盘格图案保存为图像文件
cv2.imwrite("chessboard.png", chessboard)

# 将图像插入PDF文件
pdf_filename = "chessboard.pdf"
c = canvas.Canvas(pdf_filename, pagesize=A4)
width, height = A4

# 将图像转换为合适的尺寸以适应A4纸
chessboard_img = cv2.cvtColor(chessboard, cv2.COLOR_GRAY2RGB)
image_filename = "chessboard_resized.png"
cv2.imwrite(image_filename, chessboard_img)

# 添加图像到PDF
c.drawImage(image_filename, 0, 0, width, height, preserveAspectRatio=True)
c.showPage()
c.save()

print(f"标定板已保存为 {pdf_filename}")
