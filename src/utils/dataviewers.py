
import torch
import cv2
from datetime import timedelta
import numpy as np
import sys
import dv_processing as dv

# if sys.version_info.major != 3 or sys.version_info.minor < 12:
#     import dv_processing as dv
# else:
    
sys.path.append("/usr/lib/python3/dist-packages")
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_base import EventCDBuffer

class dataviewer:
    def __init__(self, camera):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_events = 1000
        self.width, self.height = None, None
        self.events = None
        self.instant_events = None
        self.window_name = "Event Frame"
        self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.model = None
        self.slicer = None
        self.filter = None
        self.reader = None
        self.events_history = [torch.zeros(1, 4, dtype=torch.float32, device=self.device) for _ in range(1)]  # History of 5 frames
        self.camera = camera
    def retrieveEvents(self, events):
        self.instant_events = events
    def setModel(self, model):
        self.model = model
        self.model.width = self.width
        self.model.height = self.height
        self.model.eval()
        self.model.to(self.device)
    def extractEvents(self, events, reversex = False):
        xs = self.width - events["x"] -1 if reversex else events["x"]
        ys = events["y"]
        ps = 2 * events["polarity"] - 1 if reversex else  2 *events["p"] - 1
        
        ts = events["timestamp"] if reversex else  events["t"]
        ts = ts - ts.min()  # Normalize timestamps to start from 0
        events_tensor = torch.stack(( torch.tensor(ts.copy()), torch.tensor(xs.copy()).float(),
                                            torch.tensor(ys.copy()).float(), 
                                            torch.tensor(ps.copy()).float()), dim=1)  # [N, 4].long()
       

        return events_tensor.to(self.device)
    
    def predict(self):
        self.events_history[0:-1] = self.events_history[1:]  # Shift history
        self.events_history[-1] = self.events
        events_tensor = torch.concat(self.events_history, dim=0)
        seq_events = events_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        predictions, encodings, seq_events = self.model(seq_events)
        return predictions, seq_events

    def mergePredictions(self, img, predictions):   
        
        pred = predictions[0, 0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.5, pred, 0.5, 0)
        
        # resize twice the size
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        return img
    def showImage(self, img):
        
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            cv2.destroyAllWindows()
            exit(0)
    def processEvents(self, events, reversex=False):

        events_tensor = self.extractEvents(events.numpy().copy(), reversex=reversex)
        self.events = events_tensor
        predictions, seq_events = self.predict()
        self.predictions = predictions.clone()

        img = np.sum(seq_events[0][0].detach().cpu().numpy(), axis=0).astype(np.uint8)
        img[img!=0] = 255  # Set non-zero pixels to white
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        merged_img = self.mergePredictions(img, predictions)
        self.showImage(merged_img)
    def run(self):
        raise NotImplementedError("This method should be implemented in subclasses")
def _refractory_filter_numpy(ev_np, ref_us, height, width):
    # ev_np fields typically: ['timestamp','x','y','polarity']
    t_key = 'timestamp' if 'timestamp' in ev_np.dtype.names else 't'
    last_t = np.full((height, width), -1e18, dtype=np.float64)
    keep = np.zeros(ev_np.shape[0], dtype=bool)

    t = ev_np[t_key].astype(np.float64)
    x = ev_np['x'].astype(np.int32)
    y = ev_np['y'].astype(np.int32)
    for i in range(ev_np.shape[0]):
        ti, xi, yi = t[i], x[i], y[i]
        if ti - last_t[yi, xi] >= ref_us:
            keep[i] = True
            last_t[yi, xi] = ti
    return ev_np[keep]
class dataviewerdavis(dataviewer): ## Davis
    def __init__(self, camera):
        print("Using dv_processing for event processing")
        super().__init__(camera)

        self.width, self.height = self.camera.getEventResolution()
        self.max_events = 50000  # set to the training target (e.g., 1000/2000)

        self.slicer = dv.EventStreamSlicer()
        # Prefer slicing by count to stabilize input distribution
        # try:
            # if available in your dv version
            # self.slicer.doEveryNumberOfEvents(self.max_events, self.retrieveEvents)
            # self.slice_by_count = True
        # except Exception:
            # fallback to time slices (make them short)
        from datetime import timedelta
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=100), self.retrieveEvents)
        self.slice_by_count = False

        # Keep BA filter but it won’t remove hot pixels/overactive rows by itself
        from datetime import timedelta
        self.filter = dv.noise.BackgroundActivityNoiseFilter(
            (self.width, self.height),
            backgroundActivityDuration=timedelta(milliseconds=10)
        )

    def run(self):
        while self.camera.isRunning():
            self.instant_events = None
            events = self.camera.getNextEventBatch()
            if events is None or len(events) == 0:
                continue

            self.slicer.accept(events)
            if self.instant_events is None or len(self.instant_events) == 0:
                continue

            ev = self.instant_events
            try:
                ev_np = ev.numpy()
                # cap by count
                if ev_np.shape[0] > self.max_events * 4:  # cap early to keep runtime OK
                    ev_np = ev_np[-self.max_events * 4:]

                # refractory 1000 us (1 ms) — tune for your scene
                ev_np = _refractory_filter_numpy(ev_np, ref_us=1000, height=self.height, width=self.width)

                # final cap to training target
                if ev_np.shape[0] > self.max_events:
                    ev_np = ev_np[-self.max_events:]

                ev = dv.EventStore.fromNumpy(ev_np)
            except Exception:
                pass

            self.filter.accept(ev)
            filtered_events = self.filter.generateEvents()
            if filtered_events is None or len(filtered_events) == 0:
                continue
            self.processEvents(filtered_events, reversex=True)
           
            
            
class dataviewerprophesee(dataviewer):  ## prophesee
    def __init__(self, camera):
        
        super().__init__(camera)
        
        print("Using metavision_sdk_stream for event processing")
        self.width, self.height = self.camera.width(), self.camera.height()
        slice_condition = SliceCondition.make_n_us(100000)
        self.slicer = CameraStreamSlicer(self.camera.move(), slice_condition=slice_condition)
        self.activity_filter = ActivityNoiseFilterAlgorithm(self.width, self.height, 20000)
    def run(self):
        
        for slice in self.slicer:
            events_buf = EventCDBuffer()
            self.activity_filter.process_events(slice.events, events_buf)
            
            self.processEvents(events_buf, reversex=False)
            