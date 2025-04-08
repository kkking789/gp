import time

from track.track_node import TrackThread
from sensor.sensor_node import SensorThread
from yolo.yolo_node import DetectThread
from brain.brain_node import BrainThread

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
