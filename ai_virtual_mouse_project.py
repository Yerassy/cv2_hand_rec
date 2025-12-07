import cv2
import numpy as np
import time
import HandTrackingModule as htm
import pyautogui
import math
from collections import deque

# Инициализация видеозахвата
wCam, hCam = 640, 480
frameR = 100
smoothening = 5

# Настройка сглаживания курсора
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Очередь для сглаживания движений
x_history = deque(maxlen=smoothening)
y_history = deque(maxlen=smoothening)

# Получение разрешения экрана
wScr, hScr = pyautogui.size()

# Инициализация детектора рук
detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)

# Настройка видеозахвата - используем камеру 0
print("🔄 Подключаемся к камере...")
cap = cv2.VideoCapture(0)

# Устанавливаем параметры камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS

# Даем камере время на инициализацию
time.sleep(2)

# Проверяем, открыта ли камера
if not cap.isOpened():
    print("❌ Ошибка: Не удалось открыть камеру!")
    print("Возможные решения:")
    print("1. Закройте все программы, использующие камеру (Zoom, FaceTime и т.д.)")
    print("2. Дайте разрешение терминалу на использование камеры:")
    print("   Системные настройки → Конфиденциальность и безопасность → Камера")
    print("3. Перезагрузите Mac и попробуйте снова")
    exit(1)

print("✅ Камера успешно подключена!")

# Переменные для расчета FPS
pTime = 0
click_threshold = 40
click_counter = 0
click_delay = 15

print("=" * 50)
print("🎮 AI Virtual Mouse запущен! 🎮")
print("=" * 50)
print(f"📺 Разрешение экрана: {wScr}x{hScr}")
print("👆 Управление:")
print("   - Поднимите УКАЗАТЕЛЬНЫЙ палец: перемещение курсора")
print("   - Поднимите УКАЗАТЕЛЬНЫЙ и СРЕДНИЙ пальцы: левый клик")
print("   - Нажмите 'Q' для выхода")
print("   - Нажмите '+'/- для увеличения/уменьшения порога клика")
print("=" * 50)

# Отключаем защиту pyautogui
pyautogui.FAILSAFE = False

frame_count = 0
last_hand_detected = False

while True:
    # 1. Чтение кадра с камеры
    success, img = cap.read()
    frame_count += 1
    
    if not success:
        print("⚠️ Не удалось получить кадр с камеры!")
        if frame_count % 50 == 0:  # Сообщаем каждые 50 кадров
            print("Проверьте камеру...")
        continue
    
    # 2. Зеркальное отображение для интуитивного управления
    img = cv2.flip(img, 1)
    
    # 3. Обнаружение руки и ориентиров
    img = detector.findHands(img, draw=True)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    hand_detected = len(lmList) != 0
    
    # 4. Если найдена рука
    if hand_detected:
        if not last_hand_detected:
            print("✅ Рука обнаружена!")
        last_hand_detected = True
        
        # Получаем координаты кончиков указательного и среднего пальцев
        try:
            x1, y1 = lmList[8][1], lmList[8][2]  # Кончик указательного пальца
            x2, y2 = lmList[12][1], lmList[12][2]  # Кончик среднего пальца
        except IndexError:
            print("⚠️ Не все точки руки обнаружены, продолжаем...")
            continue
        
        # 5. Проверяем, какие пальцы подняты
        fingers = detector.fingersUp()
        
        # 6. Режим перемещения курсора (поднят только указательный палец)
        if fingers[1] == 1 and fingers[2] == 0:
            # Преобразуем координаты в координаты экрана
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            
            # Сглаживание движений
            x_history.append(x3)
            y_history.append(y3)
            
            if len(x_history) > 0:
                x3_smooth = sum(x_history) / len(x_history)
                y3_smooth = sum(y_history) / len(y_history)
            else:
                x3_smooth, y3_smooth = x3, y3
            
            # Перемещаем курсор
            try:
                clocX = plocX + (x3_smooth - plocX) / smoothening
                clocY = plocY + (y3_smooth - plocY) / smoothening
                
                clocX = max(0, min(wScr, clocX))
                clocY = max(0, min(hScr, clocY))
                
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                
                # Рисуем круг на кончике указательного пальца
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "MOVE", (x1 - 30, y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            except Exception as e:
                if frame_count % 100 == 0:
                    print(f"⚠️ Ошибка при перемещении курсора: {e}")
        
        # 7. Режим клика (подняты указательный и средний пальцы)
        elif fingers[1] == 1 and fingers[2] == 1:
            # Рассчитываем расстояние между пальцами
            length, img, info = detector.findDistance(8, 12, img, draw=True)
            
            # Отображаем расстояние
            cv2.putText(img, f"Dist: {int(length)}", (info[4] - 20, info[5] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Если расстояние меньше порога - выполняем клик
            if length < click_threshold:
                cv2.circle(img, (info[4], info[5]), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "CLICK", (info[4] - 30, info[5] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Защита от двойных кликов
                if click_counter == 0:
                    try:
                        pyautogui.click()
                        print(f"✅ Клик выполнен! (расстояние: {int(length)})")
                        click_counter = click_delay
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"❌ Ошибка при клике: {e}")
            else:
                cv2.circle(img, (info[4], info[5]), 15, (255, 255, 0), cv2.FILLED)
                cv2.putText(img, "READY", (info[4] - 30, info[5] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 8. Уменьшаем счетчик кликов
        if click_counter > 0:
            click_counter -= 1
        
        # 9. Рисуем ограничивающую область
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                     (255, 0, 255), 2)
    else:
        if last_hand_detected:
            print("⏸️ Рука не обнаружена")
            last_hand_detected = False
    
    # 10. Расчет и отображение FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    # Отображение информации
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f'Threshold: {click_threshold}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(img, 'AI Virtual Mouse', (10, hCam - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f'Frame: {frame_count}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Отображение инструкций
    cv2.putText(img, 'Index: Move', (wCam - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, 'Index+Middle: Click', (wCam - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, 'Q: Quit', (wCam - 150, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, '+/-: Adjust threshold', (wCam - 150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 11. Отображение изображения
    cv2.imshow("🎮 AI Virtual Mouse", img)
    
    # 12. Обработка клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🛑 Завершение работы...")
        break
    elif key == ord('+'):
        click_threshold = min(100, click_threshold + 5)
        print(f"➕ Порог клика увеличен: {click_threshold}")
    elif key == ord('-'):
        click_threshold = max(10, click_threshold - 5)
        print(f"➖ Порог клика уменьшен: {click_threshold}")

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
print("=" * 50)
print(f"✅ Программа завершена успешно!")
print(f"📊 Всего обработано кадров: {frame_count}")
print("=" * 50)