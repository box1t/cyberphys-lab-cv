import os
import shutil
import yaml

# пути
base_path = "/home/snowwy/Downloads/archive/rice"
images_dir = os.path.join(base_path, "images/train")
labels_dir = os.path.join(base_path, "labels/train")
yaml_path = os.path.join(base_path, "data.yaml")

output_dir = "/home/snowwy/data/train"

# читаем yaml
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# классы
classes = {str(i): name for i, name in enumerate(data['names'])}

# создаём папки
for cls_name in classes.values():
    os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

# обработка
for file in os.listdir(images_dir):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(images_dir, file)
    label_path = os.path.join(labels_dir, file.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        lines = f.readlines()

    if len(lines) == 0:
        continue

    # берём первый объект
    class_id = lines[0].split()[0]

    if class_id not in classes:
        print(f"Unknown class_id: {class_id}")
        continue

    class_name = classes[class_id]

    shutil.copy(img_path, os.path.join(output_dir, class_name, file))