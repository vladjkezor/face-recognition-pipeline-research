import os
import gdown
import zipfile

def load_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_dir = os.path.join(project_root, 'data', 'celeba')
    zip_path = os.path.join(data_dir, 'img.zip')
    img_dir = os.path.join(data_dir, 'img')

    os.makedirs(img_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        print('Идет скачивание архива')
        gdown.download('https://drive.google.com/uc?id=1Kh9aYL7hehGDQpy8EuJzvDFxoyOikHeM', zip_path)
    else:
        print('Архив  скачан')
    if not os.listdir(img_dir):  
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print('Распаковка архива')
            zip_ref.extractall(data_dir)
    else:
        print('Архив распакован')


if __name__ == '__main__':
    load_data()
