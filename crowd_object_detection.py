import cv2
import numpy as np
import pandas as pd
import imutils
import os
import matplotlib.pyplot as plt

# Деплоємо нашу модель та параметри
protopath = "MobileNetSSD_deploy.prototxt.txt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Задання класів для обробки зображень
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Завантаження датасету зображень
dataset_path = 'crowd_dataset/frames/frames'
image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
               os.path.isfile(os.path.join(dataset_path, f))]

# Створення пустого DataFrame для результатів
results_df = pd.DataFrame(columns=['Image', 'Predicted', 'Count'])


# Функція що перетворює числовий ідентифікатор зображення
# у відносний шлях до файлу, заповнюючи початкові нулі
# та додаючи розширення і каталог файлу.
def replace_zeros(image_id: int) -> str:
    image_id = str(image_id).rjust(6, '0')
    return f'crowd_dataset/frames/frames/seq_{image_id}.jpg'


# Завантаження labels.csv файлу
labels_file = 'crowd_dataset/labels.csv'

# Вивід даних з датасету
data = pd.read_csv(labels_file)
data['path'] = data['id'].apply(replace_zeros)
print("General output information of the dataset in the amount of 5 initial and 5 final images:")
print(data.head(5))
print(data.tail(5))

# Вивід статистики даних датасету
stats = data.describe()
print("\nOutput of computed dataset statistics:")
print(stats)

# Виводимо статистику датасету за допомогою бібліотеки Matplotlib
plt.hist(data['count'], bins=20)
plt.axvline(stats.loc['mean', 'count'], label='Mean value', color='yellow')
plt.legend()
plt.xlabel('Number of people')
plt.ylabel('Frequency of the number of people')
plt.title('Histogram of the number of silhouette detections')
plt.show()

print("\nClassification of objects has been started, please wait for results...")

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_size = imutils.resize(image, width=640)

    (H, W) = image_size.shape[:2]

    blob = cv2.dnn.blobFromImage(image_size, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()

    count = 0

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                count = count + 1

    # Отримати ground truth кількість людей з датасету
    image_filename = os.path.basename(image_path)
    image_id = int(os.path.splitext(image_filename)[0].split('_')[1])
    truth_count = data[data['id'] == image_id]['count'].values[0]

    # Додати результати до DataFrame
    results_df = pd.concat([results_df, pd.DataFrame(
        {'Image': [image_path], 'Predicted': [count], 'Count': [truth_count]})],
                           ignore_index=True)

results_df['MAE'] = (results_df['Count'] - results_df['Predicted']).abs()
results_df['MSE'] = results_df['mae'] ** 2

plt.hist(results_df['mae'], bins=20)
plt.xlabel('Absolute Errors')
plt.ylabel('Errors frequency')
plt.title('Histogram of Absolute Errors')
plt.show()

plt.scatter(results_df['Count'], results_df['Predicted'])
plt.xlabel('Actual person count')
plt.ylabel('Predicted person count')
plt.title('Predicted vs Actual Count')
plt.show()

# Збереження результатів обробки зображень у файл .csv
results_df.to_csv('results.csv', index=False)

# Вивести результати
print("\nClassification has been completed:")
print(results_df)

# Обчислити метрики порівняння
diff = results_df['Predicted'] - results_df['Count']
mae = np.abs(diff).mean()
mse = (diff ** 2).mean()
rmse = np.sqrt(mse)

print("\nComparison of the results:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Square Error (RMSE): {rmse}")
