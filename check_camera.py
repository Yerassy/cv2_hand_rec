import cv2

# Проверим все доступные камеры
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Камера {i}: Доступна")
        cap.release()
    else:
        print(f"Камера {i}: Недоступна")

# Также проверим через cv2
print("\nПроверка через cv2:")
print(f"Версия OpenCV: {cv2.__version__}")