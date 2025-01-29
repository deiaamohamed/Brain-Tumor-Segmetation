import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms

# (preparation)
def CustomCrop(img):
    return transforms.functional.crop(img, 0, 0, 423, 326)

transform = transforms.Compose([
    transforms.Lambda(CustomCrop),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


brain_dir = r'E:\brain-tumor-mri-dataset\Brain-NoBrain\Brain'
no_brain_dir = r'E:\brain-tumor-mri-dataset\Brain-NoBrain\NoBrain'
train_dir = r'E:\brain-tumor-mri-dataset\Brain-NoBrain\train'
val_dir = r'E:\brain-tumor-mri-dataset\Brain-NoBrain\val'


os.makedirs(os.path.join(train_dir, 'Brain'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'No_Brain'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'Brain'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'No_Brain'), exist_ok=True)

# تعديل الصور
def apply_transforms_and_save(src_dir, dst_dir, files):
    for file in files:
        img_path = os.path.join(src_dir, file)
        img = Image.open(img_path)
        transformed_img = transform(img)
        transformed_img_pil = transforms.ToPILImage()(transformed_img)
        transformed_img_path = os.path.join(dst_dir, file)
        transformed_img_pil.save(transformed_img_path)


def split_and_move_files(src_dir, train_dst, val_dst, test_size=0.2):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)

    
    apply_transforms_and_save(src_dir, train_dst, train_files)
    apply_transforms_and_save(src_dir, val_dst, val_files)


split_and_move_files(brain_dir, os.path.join(train_dir, 'Brain'), os.path.join(val_dir, 'Brain'))


split_and_move_files(no_brain_dir, os.path.join(train_dir, 'No_Brain'), os.path.join(val_dir, 'No_Brain'))

print("Dataset split into training and validation sets with transformations applied.")