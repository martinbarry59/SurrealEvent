
import torch
import cv2
from datetime import timedelta
import numpy as np
import sys
if sys.version_info.major != 3 or sys.version_info.minor < 12:
    import dv_processing as dv
else:
    
    sys.path.append("/usr/lib/python3/dist-packages")
    from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_base import EventCDBuffer

class dataviewer:
    def __init__(self):
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
    def retrieveEvents(self, events):
        self.instant_events = events
    def setModel(self, model):
        self.model = model
        self.model.width = self.width
        self.model.height = self.height
        self.model.eval()
        self.model.to(self.device)
    def extractEvents(self, reversex = False):

        xs = self.width - self.events["x"] -1 if reversex else self.events["x"]
        ys = self.events["y"]
        ps = 2 * self.events["polarity"] - 1 if reversex else  2 *self.events["p"] - 1
        ts = self.events["timestamp"] if reversex else  self.events["t"] *1e-6
        events_tensor = torch.stack(( torch.tensor(ts.copy()).float(), torch.tensor(xs.copy() / self.width).float(),
                                            torch.tensor(ys.copy() / self.height).float(), 
                                            torch.tensor(ps.copy()).float()), dim=1)  # [N, 4].long()
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[ys[ps == 1], xs[ps == 1]] = (255, 255, 255)  # White for polarity 1
        img[ys[ps == 0], xs[ps == 0]] = (255, 0, 0)      # Blue for polarity 0
        
     
        return events_tensor, img
    def selectEvents(self, filtered_events):
        filtered_events = filtered_events.numpy().copy()
        if self.events is None:
            self.events = filtered_events.copy()
        else:
            self.events = filtered_events
    def predict(self,events_tensor):
        mask = torch.ones((events_tensor.shape[0],), dtype=torch.bool)
        seq_events = events_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        seq_masks = mask.unsqueeze(0).unsqueeze(0).to(self.device)
        predictions, encodings = self.model(seq_events, seq_masks)
        
        
        return predictions
        ## crop image to 246, 346
        # merge the predictions into the image
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
    def run(self):
        raise NotImplementedError("This method should be implemented in subclasses")
class dataviewer39(dataviewer): ## Davis
    def __init__(self):
        print("Using dv_processing for event processing")
        super().__init__()
        
        self.reader = dv.io.camera.open()
        
        self.width, self.height = self.reader.getEventResolution()

        self.slicer = dv.EventStreamSlicer()
        self.filter = dv.noise.BackgroundActivityNoiseFilter((self.width, self.height), 
                                                        backgroundActivityDuration=timedelta(milliseconds=10))
        
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=60),self.retrieveEvents)
    
    # def generateEvents(self):
    #     return self.events
    def run(self):

        while self.reader.isRunning():
            self.instant_events = None

    # Read batch of events
            events = self.reader.getNextEventBatch()
            print(events)
            if events is None or len(events)== 0:
                continue

            self.slicer.accept(events)
            if self.instant_events is None or len(self.instant_events) == 0:
                continue
            self.filter.accept(self.instant_events)
            filtered_events = self.instant_events#self.filter.generateEvents()

            # In your event processing loop:
            
            # Create a black image
            
            # Draw events: polarity 1 as white, 0 as blue (or any color you like)
            self.selectEvents(filtered_events)
            events_tensor, img = self.extractEvents(reversex=True)
            ## from float 32 to float 16
            predictions = self.predict(events_tensor)
            merged_img = self.mergePredictions(img, predictions)
            self.showImage(merged_img)
class dataviewer312(dataviewer):  ## prophesee
    def __init__(self):
        super().__init__()
        
        print("Using metavision_sdk_stream for event processing")
        self.camera = Camera.from_first_available()
        self.width, self.height = self.camera.width(), self.camera.height()
        slice_condition = SliceCondition.make_n_us(100000)
        self.slicer = CameraStreamSlicer(self.camera.move(), slice_condition=slice_condition)
        self.activity_filter = ActivityNoiseFilterAlgorithm(self.width, self.height, 20000)
    def run(self):
        
        
        for slice in self.slicer:
            events_buf = EventCDBuffer()
            self.activity_filter.process_events(slice.events, events_buf)

            self.selectEvents(events_buf)
            if len(self.events)  == 0:
                continue
            events_tensor, img = self.extractEvents(reversex=False)
            ## from float 32 to float 16
            predictions = self.predict(events_tensor)
            merged_img = self.mergePredictions(img, predictions)
            self.showImage(merged_img)