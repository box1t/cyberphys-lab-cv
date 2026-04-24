
# Инструкция по запуску

```sh
git clone <your_repo_url>
cd <project_folder>
```

```sh
pip install torch torchvision scikit-learn tqdm
```

## Конвертация датасета
Исходный датасет имеет формат YOLO, для работы с классификацией необходимо преобразовать его.
С этой целью написан запускаемый скрипт:
```sh
python3 dataset_converter.py
```

После этого появится структура:
```
/home/snowwy/data/train/
 ├── Rice__BacterialLeafBlight/
 ├── Rice__BrownSpot/
 ├── ...
```

## Запуск обучения

```sh
python3 models_framework.py
```

### 🧪 Что происходит при запуске

Скрипт автоматически:

🔹 1. Обучает baseline модель
ResNet без аугментаций
🔹 2. Обучает улучшенный бейзлайн
ResNet с аугментациями
🔹 3. Сравнивает модели

Обучаются:

ResNet
MobileNet
EfficientNet
VGG
AlexNet
SimpleCNN (собственная)
🔹 4. Обучает трансформер
Vision Transformer (облегчённая версия)
### 📊 Выходные данные

После выполнения выводится:

### 🧠 Особенности реализации
Используется transfer learning
Реализован random train/val split
Применяются data augmentation
Трансформер оптимизирован (заморозка слоёв)
