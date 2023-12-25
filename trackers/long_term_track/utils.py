import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

from submodule.LTM.core.motion_evaluator import MotionEvaluator
from submodule.PBA.core.pos_cost_assa_evaluator import AssaEvaluator


def get_motion_model(motion_model_cfg, motion_epoch):
    model_loader = MotionEvaluator(motion_model_cfg, motion_epoch)
    motion_model = model_loader.model
    return motion_model


def get_assa_model(assa_model_cfg, assa_epoch):
    model_loader = AssaEvaluator(assa_model_cfg, assa_epoch)
    assa_model = model_loader.model
    return assa_model


def relative_position(bboxes):
    """
        :param bboxes: ndarray[N × xyxy]
        :return:
    """
    rel_k = 6
    bboxes_iou = iou_similarity(bboxes, bboxes)
    row, col = np.diag_indices_from(bboxes_iou)
    bboxes_iou[row, col] = 0
    iou_mask = (bboxes_iou > 0).astype(np.int32)

    bboxes_1 = np.repeat(np.expand_dims(bboxes, 0), bboxes.shape[0], axis=0)  # [N, N, 4]
    bboxes_2 = np.repeat(np.expand_dims(bboxes, 1), bboxes.shape[0], axis=1)  # [N, N, 4]
    rel_distance = bboxes_2 - bboxes_1
    rel_distance = rel_distance * np.repeat(np.expand_dims(iou_mask, -1), 4, axis=-1)

    # rel_distance = rel_distance.sum(1)
    abs_rel_distance = np.abs(rel_distance).sum(-1)  # [N, N]
    abs_rel_distance[abs_rel_distance == 0] = 50000
    sorted_indices = np.argsort(abs_rel_distance, axis=-1)
    for i in range(rel_distance.shape[0]):
        rel_distance[i] = rel_distance[i][sorted_indices[i]]
    rel_distance = rel_distance[:, :rel_k, :]
    if rel_distance.shape[1] < rel_k:
        padding = np.zeros((rel_distance.shape[0], rel_k - rel_distance.shape[1], 4))
        rel_distance = np.concatenate((rel_distance, padding), axis=1)
    # rel_distance[rel_distance == 0] = 50000

    return rel_distance


def iou_similarity(track_bbox, det_bbox):
    """
    Compute cost based on IoU
    :type track_bbox: ndarray: (M, 4), tlbr
    :type det_bbox: ndarray: (N, 4)], tlbr

    :rtype simi_iou np.ndarray
    """
    simi_iou = np.zeros((len(track_bbox), len(det_bbox)), dtype=np.float)
    if simi_iou.size == 0:
        return simi_iou

    simi_iou = bbox_ious(
        np.ascontiguousarray(track_bbox, dtype=np.float),
        np.ascontiguousarray(det_bbox, dtype=np.float)
    )

    return simi_iou


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def oai(detections, det_idx, tracks, o_max=0.55):
    """

    :param det_idx: ndarray, indices of unmatched_dets
    :param detections: ndarray[N, xyxys]
    :param tracks: List[KalmanBoxTracker]
    :param o_max:
    :return:
    """
    if len(det_idx) == 0 or len(tracks) == 0:
        return []
    track_bbox = []
    for trk in tracks:
        if trk.last_observation.sum() < 0:
            track_bbox.append(trk.sequence[0, -1].numpy().copy())  # last observation
        else:
            track_bbox.append(trk.pred[0].numpy().copy())
    track_bbox = np.asarray(track_bbox)
    track_bbox[:, 2:] += track_bbox[:, :2]
    det_bbox = detections[det_idx][:, :4]
    iou_matrix = iou_batch(det_bbox, track_bbox)
    iou_matrix = np.max(iou_matrix, axis=1)
    rm_det_idx = det_idx[iou_matrix > o_max]
    rm_det_idx_idx = np.where(iou_matrix > o_max)
    return rm_det_idx_idx


def calibrate_idx(matched, unmatched_dets, unmatched_trks, det_indices, trk_indices):
    new_unmatched_dets = unmatched_dets if len(unmatched_dets) == 0 else det_indices[unmatched_dets]
    new_unmatched_trks = unmatched_trks if len(unmatched_trks) == 0 else trk_indices[unmatched_trks]
    new_matched = []
    for d_i, t_j in matched:
        det_idx = det_indices[d_i]
        trk_idx = trk_indices[t_j]
        new_matched.append((det_idx, trk_idx))
    new_matched = np.asarray(new_matched)
    return new_matched.astype('int64'), new_unmatched_dets.astype('int64'), new_unmatched_trks.astype('int64')


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def compute_aw_new_metric(emb_cost, w_association_emb, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus
