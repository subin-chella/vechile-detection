# Minimal Sort tracker placeholder so main.py runs.
import numpy as np

class Sort:
    def __init__(self):
        self.next_id = 0
        self.tracks = []  # each: [x1,y1,x2,y2,id]

    def update(self, detections):
        # naive: assign new id each detection each frame
        results = []
        for det in detections:
            x1,y1,x2,y2,score = det
            results.append([x1,y1,x2,y2,self.next_id])
            self.next_id += 1
        self.tracks = results
        return np.array(results)
