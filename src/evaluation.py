import numpy as np
import cv2

# My libraries
from .util import munkres as munkres
from .util import utilities as util_

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def boundary_overlap(predicted_mask, gt_mask, bound_th=0.003):
    """
    Compute true positives of overlapped masks, using dilated boundaries

    Arguments:
        predicted_mask  (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        overlap (float): IoU overlap of boundaries
    """
    assert np.atleast_3d(predicted_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(predicted_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = util_.seg2bmap(predicted_mask);
    gt_boundary = util_.seg2bmap(gt_mask);

    from skimage.morphology import disk

    # Dilate segmentation boundaries
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix), iterations=1)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix), iterations=1)

    # Get the intersection (true positives). Calculate true positives differently for
    #   precision and recall since we have to dilated the boundaries
    fg_match = np.logical_and(fg_boundary, gt_dil)
    gt_match = np.logical_and(gt_boundary, fg_dil)

    # Return precision_tps, recall_tps (tps = true positives)
    return np.sum(fg_match), np.sum(gt_match)

# This function is modeled off of P/R/F measure as described by Dave et al. (arXiv19)
def multilabel_metrics(prediction, gt, obj_detect_threshold=0.75):
    """ Compute Overlap and Boundary Precision, Recall, F-measure
        Also compute #objects detected, #confident objects detected, #GT objects.

        It computes these measures only of objects (2+), not background (0) / table (1).
        Uses the Hungarian algorithm to match predicted masks with ground truth masks.

        A "confident object" is an object that is predicted with more than 0.75 F-measure

        @param gt: a [H x W] numpy.ndarray with ground truth masks
        @param prediction: a [H x W] numpy.ndarray with predicted masks

        @return: a dictionary with the metrics
    """

    ### Compute F-measure, True Positive matrices ###

    # Get unique OBJECT labels from GT and prediction
    labels_gt = np.unique(gt)
    labels_gt = labels_gt[~np.isin(labels_gt, [BACKGROUND_LABEL, TABLE_LABEL])]
    num_labels_gt = labels_gt.shape[0]

    labels_pred = np.unique(prediction)
    labels_pred = labels_pred[~np.isin(labels_pred, [BACKGROUND_LABEL, TABLE_LABEL])]
    num_labels_pred = labels_pred.shape[0]

    # F-measure, True Positives, Boundary stuff
    F = np.zeros((num_labels_gt, num_labels_pred))
    true_positives = np.zeros((num_labels_gt, num_labels_pred))
    boundary_stuff = np.zeros((num_labels_gt, num_labels_pred, 2)) 
    # Each item of "boundary_stuff" contains: precision true positives, recall true positives

    # Edge cases
    if (num_labels_pred == 0 and num_labels_gt > 0 ): # all false negatives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 1.,
                'Objects Recall' : 0.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 0.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred > 0 and num_labels_gt == 0 ): # all false positives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 0.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 0.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred == 0 and num_labels_gt == 0 ): # correctly predicted nothing
        return {'Objects F-measure' : 1.,
                'Objects Precision' : 1.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 1.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 1.,
                }

    # For every pair of GT label vs. predicted label, calculate stuff
    for i, gt_i in enumerate(labels_gt):

        gt_i_mask = (gt == gt_i)

        for j, pred_j in enumerate(labels_pred):
            
            pred_j_mask = (prediction == pred_j)
            
            ### Overlap Stuff ###

            # true positive
            A = np.logical_and(pred_j_mask, gt_i_mask)
            tp = np.int64(np.count_nonzero(A)) # Cast this to numpy.int64 so 0/0 = nan
            true_positives[i,j] = tp 
            
            # precision
            prec = tp/np.count_nonzero(pred_j_mask)
            
            # recall
            rec = tp/np.count_nonzero(gt_i_mask)
            
            # F-measure
            F[i,j] = (2 * prec * rec) / (prec + rec)

            ### Boundary Stuff ###
            boundary_stuff[i,j] = boundary_overlap(pred_j_mask, gt_i_mask)

    ### More Boundary Stuff ###
    boundary_prec_denom = 0. # precision_tps + precision_fps
    for pred_j in labels_pred:
        pred_mask = (prediction == pred_j)
        boundary_prec_denom += np.sum(util_.seg2bmap(pred_mask))
    boundary_rec_denom = 0. # recall_tps + recall_fns
    for gt_i in labels_gt:
        gt_mask = (gt == gt_i)
        boundary_rec_denom += np.sum(util_.seg2bmap(gt_mask))


    ### Compute the Hungarian assignment ###
    F[np.isnan(F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(F.max() - F.copy()) # list of (y,x) indices into F (these are the matchings)
    idx = tuple(np.array(assignments).T)

    ### Compute the number of "detected objects" ###
    num_obj_detected = 0
    for a in assignments:
        if F[a] > obj_detect_threshold:
            num_obj_detected += 1

    # Overlap measures
    precision = np.sum(true_positives[idx]) / np.sum(prediction.clip(0,2) == OBJECTS_LABEL)
    recall = np.sum(true_positives[idx]) / np.sum(gt.clip(0,2) == OBJECTS_LABEL)
    F_measure = (2 * precision * recall) / (precision + recall)
    if np.isnan(F_measure): # b/c precision = recall = 0
        F_measure = 0

    # Boundary measures
    boundary_precision = np.sum(boundary_stuff[idx][:,0]) / boundary_prec_denom
    boundary_recall = np.sum(boundary_stuff[idx][:,1]) / boundary_rec_denom
    boundary_F_measure = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
    if np.isnan(boundary_F_measure): # b/c/ precision = recall = 0
        boundary_F_measure = 0


    return {'Objects F-measure' : F_measure,
            'Objects Precision' : precision,
            'Objects Recall' : recall,
            'Boundary F-measure' : boundary_F_measure,
            'Boundary Precision' : boundary_precision,
            'Boundary Recall' : boundary_recall,
            'obj_detected' : num_labels_pred,
            'obj_detected_075' : num_obj_detected,
            'obj_gt' : num_labels_gt,
            'obj_detected_075_percentage' : num_obj_detected / num_labels_gt,
            }
