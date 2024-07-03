import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# 定义圆圈标定板的大小
rows = 11  # 圆点行数
cols = 4  # 圆点列数
circle_diameter = 5  # 圆圈的直径（毫米）
circle_radius = circle_diameter // 2
spacing = 10  # 圆圈间的间距（毫米）

# 读取图像文件并寻找圆点
image = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)
pattern_size = (cols, rows)
found, centers = cv2.findCirclesGrid(image, pattern_size, cv2.CALIB_CB_ASYMMETRIC_GRID)

# 显示结果
if found:
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(image_color, pattern_size, centers, found)
    cv2.imshow("Circle Grid Detection", image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未找到圆点标定板的中心点。")

# 打印标定板规格
print(f"标定板规格：行数={rows}, 列数={cols}, 圆点直径={circle_diameter}毫米, 圆点间距={spacing}毫米")
