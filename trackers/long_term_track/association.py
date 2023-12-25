import numpy as np
import torch

from .utils import iou_batch, relative_position, \
    linear_assignment, compute_aw_new_metric


def interaction_score(bboxes):
    """

    :param bboxes: ndarray[N × xyxy]
    :return:
    """
    bboxes_iou = iou_batch(bboxes, bboxes)
    row, col = np.diag_indices_from(bboxes_iou)
    bboxes_iou[row, col] = 0
    iou_mask = (bboxes_iou > 0).astype(np.int32)

    bboxes_1 = np.repeat(np.expand_dims(bboxes, 0), bboxes.shape[0], axis=0)  # [N, N, 4]
    bboxes_2 = np.repeat(np.expand_dims(bboxes, 1), bboxes.shape[0], axis=1)  # [N, N, 4]
    rel_distance = np.abs(bboxes_2 - bboxes_1)
    rel_distance = rel_distance * np.repeat(np.expand_dims(iou_mask, -1), 4, axis=-1)
    rel_distance = rel_distance.sum(1)
    return rel_distance


def interaction_cost(bboxes1, bboxes2):
    inter_score_1 = interaction_score(bboxes1[:, :4])  # [N, 4]
    inter_score_2 = interaction_score(bboxes2[:, :4])  # [M, 4]

    inter_score_1 = np.repeat(np.expand_dims(inter_score_1, 1), inter_score_2.shape[0], axis=1)  # [N, M, 4]
    inter_score_2 = np.repeat(np.expand_dims(inter_score_2, 0), inter_score_1.shape[0], axis=0)  # [N, M, 4]
    inter_cost = np.linalg.norm(inter_score_2 - inter_score_1, ord=2, axis=-1)  # [N, M] Euclidean distance

    _range = np.max(inter_cost) - np.min(inter_cost)
    inter_cost = (inter_cost - np.min(inter_cost)) / _range

    return inter_cost


def associate_with_inter_cost(
        detections,
        trackers,
        det_embs,
        trk_embs,
        iou_threshold,
        trk_sequences,
        assa_model,
        w_assoc_emb,
        aw_param,
        emb_off,
):
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers))
        )

    iou_matrix = iou_batch(detections, trackers)

    max_seq_len = max([len(trk) for trk in trk_sequences])
    for i in range(len(trk_sequences)):
        if len(trk_sequences[i]) < max_seq_len:
            trk_sequences[i] = np.pad(trk_sequences[i], ((max_seq_len - len(trk_sequences[i]), 0), (0, 0)), 'constant',
                                      constant_values=(0, 0))
    det_bbox = detections[:, :4]
    trk_seq = np.asarray(trk_sequences)[:, :-1, :]
    det_rel_pos = relative_position(det_bbox)
    trk_rel_pos = relative_position(trk_seq[:, -1, :])
    pos_simi = assa_model(
        torch.from_numpy(trk_seq).unsqueeze(0).to(torch.float32).cuda(),
        torch.from_numpy(trk_rel_pos).unsqueeze(0).to(torch.float32).cuda(),
        torch.from_numpy(det_bbox).unsqueeze(0).to(torch.float32).cuda(),
        torch.from_numpy(det_rel_pos).unsqueeze(0).to(torch.float32).cuda())
    pos_simi = pos_simi.view(pos_simi.shape[1], pos_simi.shape[2]).detach().t().cpu().numpy()

    emb_cost = None
    if not emb_off:
        emb_cost = None if (trk_embs.shape[0] == 0 or det_embs.shape[
            0] == 0) else det_embs @ trk_embs.T  # cosine similarity

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                pass
            if not emb_off:
                w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                emb_cost *= w_matrix
            else:
                emb_cost *= w_assoc_emb

            final_cost = -(pos_simi + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_with_iou(
    detections,
    trackers,
    det_embs,
    trk_embs,
    iou_threshold,
    w_assoc_emb,
    aw_param,
    emb_off,
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    emb_cost = None
    if not emb_off:
        emb_cost = None if (trk_embs.shape[0] == 0 or det_embs.shape[0] == 0) else det_embs @ trk_embs.T

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                pass
            w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
            emb_cost *= w_matrix

            final_cost = -(iou_matrix + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_zombie_tracks(det_embs, trk_embs, dets, trk,
                            emb_threshold=0.45, iou_threshold=0.15):
    if len(det_embs) == 0 or len(trk_embs) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(det_embs)),
            np.arange(len(trk_embs))
        )
    emb_cost = None if (trk_embs.shape[0] == 0 or det_embs.shape[
        0] == 0) else det_embs @ trk_embs.T  # cosine similarity

    iou_matrix = iou_batch(dets, trk)
    final_cost = iou_matrix + emb_cost
    matched_indices = linear_assignment(-final_cost)

    unmatched_detections = []
    for d, det in enumerate(det_embs):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trk_embs):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if emb_cost[m[0], m[1]] < emb_threshold or iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
