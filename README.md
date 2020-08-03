# kalman-multi-object-tracker
Kalman Filter Based Multiple Object Tracker. 

## Usage
The ObjectTracker class can be initalised by a list of bounding boxes, the required format is commented in the object_tracker.py file.

The ObjectTracker class has two main methods, *predict* and *update*.

### Predict
Returns a list of predicted bounding boxes for the tracked objects.

&nbsp;&nbsp;&nbsp;&nbsp;Args:

&nbsp;&nbsp;&nbsp;&nbsp;Returns:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*preds*: List of bounding box predictions 
  

### Update
Update trackers according to ground truths (gts). Ground truths are assigned to trackers according to the assignment which maximises the total intersection over union score of the predictions and ground truth bounding boxes. This assignment is found with the Hungarian Algorithm.

&nbsp;&nbsp;&nbsp;&nbsp;Args:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*gts*: List of object ground truth bounding boxes

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*preds*: List of predicted object bounding boxes

&nbsp;&nbsp;&nbsp;&nbsp;Returns:

## Shortcomings

* Tracking ids sometimes swap if objects pass infront of one another.

* Wrong assignment can occur when multiple objects leave/become occluded and enter/become unoccluded between updates.
