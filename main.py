import time

from Task1_Track.Task1 import TrackThread
from Task2_Sensor.Task2 import SensorThread
from Task3_Yolo.Task3 import DetectThread
from Task4_Brain.Task4 import BrainThread

if __name__ == '__main__':
    task1 = TrackThread()
    task2 = SensorThread()
    task3 = DetectThread()
    task4 = BrainThread()

    task1.start()
    task2.start()
    task3.start()
    task4.start()

    start_time = time.time()
    while True:
        run_time = time.time() - start_time
        if run_time >= 60 * 2:
            break
