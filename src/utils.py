import os
import gdown
import zipfile
import cv2
import torch
import numpy as np


def load_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_dir = os.path.join(project_root, 'data', 'celeba')
    zip_path = os.path.join(data_dir, 'img.zip')
    img_dir = os.path.join(data_dir, 'img')

    os.makedirs(img_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print('Идет скачивание архива')
        gdown.download('https://drive.google.com/uc?id=1Kh9aYL7hehGDQpy8EuJzvDFxoyOikHeM', zip_path)
        print('Архив скачан')
    else:
        print('Архив уже скачан')
    if not os.listdir(img_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print('Распаковка архива')
            zip_ref.extractall(data_dir)
            print('Архив распакован')
    else:
        print('Архив уже распакован')


def get_landmarks_from_heatmaps(heatmaps):
    keypoints = []
    for heatmap in heatmaps:
        idx = heatmap.view(-1).argmax()
        y, x = divmod(idx.item(), heatmap.shape[1])
        keypoints.append((x, y))
    return keypoints


def align_face(image, model, output_size=(128, 128)):

    template = np.array([
        [38, 51],
        [73, 51],
        [56, 71],
        [41, 92],
        [70, 92],
    ], dtype=np.float32)

    template *= output_size[0] / 112

    model.eval()
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(next(model.parameters()).device)
        heatmaps = model(input_tensor)[-1][0].cpu()

    keypoints = get_landmarks_from_heatmaps(heatmaps)
    src = np.array(keypoints, dtype=np.float32)

    M = cv2.estimateAffinePartial2D(src, template, method=cv2.LMEDS)[0]
    
    image = image * 0.5 + 0.5
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    aligned_face = cv2.warpAffine(img_np, M, output_size)

    return aligned_face


if __name__ == '__main__':
    load_data()
