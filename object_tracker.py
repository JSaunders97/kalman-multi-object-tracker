from kalman_filter import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment


class ObjectTracker:
    # Mulit. object tracker using kalman filter

    def __init__(self, initial_bounding_boxes):
        # initial_bounding_boxes is list of numpy arrays, with each representing a bounding box,
        # np.array([[x_top_left], [y_top_left], [width], [height], [v_x], [v_y], [v_width], [v_height]])

        self.objects = []
        self.cur_max_id = 0 # Next available id after the current maximum id
        self.available_ids = [] # Available ids less than cur_max_id
        self.missing_remove = 10 # Remove object from tracker after missing in x frames

        # add trackers for initial bounding boxes
        for bbox in initial_bounding_boxes:
            kalman = KalmanFilter(bbox, np.eye(8), 1)
            tracker = {'kf': kalman,
                       'id': self.cur_max_id,
                       'missing_counter': 0}
            self.cur_max_id += 1
            self.objects.append(tracker)

    def iou_score(self, gt, pred):

        if (gt[0] + gt[2] <= pred[0] or
            pred[0] + pred[2] <= gt[0] or
            gt[1] + gt[3] <= pred[1] or
            pred[1] + pred[3] <= gt[1]):
            return 0
        else:
            xA = max(gt[0], pred[0])
            yA = max(gt[1], pred[1])
            xB = min(gt[0] + gt[2], pred[0] + pred[2])
            yB = min(gt[1] + gt[3], pred[1] + pred[3])

            inter = (xB - xA) * (yB - yA)

            union = gt[2] * gt[3] + pred[2] * pred[3] - inter

            return inter/union

    def build_matrix(self, gts, preds):

        len_pred = len(preds)
        len_gt = len(gts)

        if len_gt < len_pred:
            matrix = np.zeros((len_pred, len_pred))
        else:
            matrix = np.zeros((len_gt, len_gt))

        for row, pred in enumerate(preds):
            for col, gt in enumerate(gts):
                matrix[row, col] = self.iou_score(gt, pred)

        return np.max(matrix) - matrix

    def match_boxes(self, gts, preds):

        cost_matrix = self.build_matrix(gts, preds)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return row_ind, col_ind

    def predict(self):
        # Predict the next state of bounding boxes

        preds = []

        for obj in self.objects:
            preds.append(obj['kf'].predict())

        return preds

    def update(self, gts, preds):
        """ 
        Update trackers according to ground truths (gts)

        Args:
            gts: object ground truth bounding boxes
            preds: predicted object bounding boxes
          
        """

        pred_ind, gt_ind = self.match_boxes(gts, preds)

        diff = len(gts) - len(preds)

        if diff > 0:
            #person(s) enters scene
            for ind in range(len(preds)):
                self.objects[pred_ind[ind]]['kf'].update(gts[gt_ind[ind]])
            self.add_objects(diff, gt_ind, gts)

        elif diff < 0:
            #person(s) leaves scene
            remove_ind = []

            for ind in range(len(preds)):

                if gt_ind[ind] >= len(gts):
                    self.objects[pred_ind[ind]]['missing_counter'] += 1
                    if self.objects[pred_ind[ind]]['missing_counter'] == self.missing_remove:
                        remove_ind.append(pred_ind[ind])

                else:
                    self.objects[pred_ind[ind]]['kf'].update(gts[gt_ind[ind]])

            self.remove_objects(remove_ind)

        else:
            #number of people unchanged
            for ind, cor in enumerate(pred_ind):
                self.objects[cor]['kf'].update(gts[gt_ind[ind]])

    def add_objects(self, num_new, gt_ind, gts):
        len_available = len(self.available_ids)

        for new_obj in range(num_new):
            bbox = np.array([[gts[gt_ind[-(new_obj+1)]][0]],[gts[gt_ind[-(new_obj+1)]][1]],
                             [gts[gt_ind[-(new_obj+1)]][2]],[gts[gt_ind[-(new_obj+1)]][3]],
                             [1],[1],[1],[1]])
            kalman = KalmanFilter(bbox, np.eye(8), 1)

            if new_obj < len_available:
                id = self.available_ids.pop()
                tracker = {'kf': kalman,
                           'id': id,
                           'missing_counter': 0}

            else:
                tracker = {'kf': kalman,
                           'id': self.cur_max_id,
                           'missing_counter': 0}
                self.cur_max_id += 1

            self.objects.append(tracker)

    def remove_objects(self, remove_ind):
        for remove in sorted(remove_ind, reverse=True):

            if self.objects[remove]['id'] + 1 == self.cur_max_id:
                self.cur_max_id -= 1
            else:
                self.available_ids.append(self.objects[remove]['id'])

            self.objects.pop(remove)
