from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans


def extract_colors(image_path, num_colors=5):
    """
    이미지에서 주요 색상을 추출하고, 각 색상의 픽셀 비중을 퍼센트로 계산합니다.

    :param image_path: 이미지 파일의 경로
    :param num_colors: 추출할 색상의 수
    :return: 추출된 색상의 RGB 값과 각 색상의 비중(퍼센트) 목록
    """
    # 이미지를 열고 RGB로 변환
    image = Image.open(image_path)
    image = image.convert('RGB')

    # 이미지를 numpy 배열로 변환
    np_image = np.array(image)
    h, w, _ = np_image.shape
    np_image = np_image.reshape((h * w, 3))

    # K-means 군집화 실행
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(np_image)

    # 중심점의 색상을 반환
    centers = kmeans.cluster_centers_.astype(int)

    # 각 군집의 픽셀 비중 계산
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    total_pixels = len(labels)
    percentages = (label_counts / total_pixels) * 100

    return list(zip(centers, percentages))

# 컬러 팔레트 이미지 생성 및 저장
def create_proportional_palette_image(palette, image_height=100):
    # 색상 비율에 따라 각 블록의 너비 계산
    total_percentage = sum(percentage for _, percentage in palette)
    blocks_width = [int((percentage / total_percentage) * 1000) for _, percentage in palette]

    # 전체 이미지의 너비는 모든 블록 너비의 합
    total_width = sum(blocks_width)

    palette_image = Image.new("RGB", (total_width, image_height))
    draw = ImageDraw.Draw(palette_image)

    x0 = 0
    for (color, _), block_width in zip(palette, blocks_width):
        x1 = x0 + block_width
        draw.rectangle([x0, 0, x1, image_height], fill=tuple(color))
        x0 = x1

    palette_image.save('color_palette.png')

# 예시 이미지 파일 경로
image_path = './plant.jfif'

# 컬러 팔레트 추출 및 퍼센트 계산
palette = extract_colors(image_path)
for color, percentage in palette:
    print(f"색상: {color}, 비중: {percentage:.2f}%")

# 컬러 팔레트 이미지 생성 및 저장
create_proportional_palette_image(palette)

##############이미지 인풋에 따라 픽셀 비중 기반 이미지 컬러 팔레트 생성########################

