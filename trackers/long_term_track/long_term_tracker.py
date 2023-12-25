import torch.nn.functional as F

from .association import *
from .cmc import CMCComputer
from .embedding import EmbeddingComputer
from .utils import get_motion_model, get_assa_model, oai, calibrate_idx, relative_position


class STrack:

    count = 0

    def __init__(self, bbox, emb=None):
        self.time_since_update = 0
        self.id = STrack.count
        STrack.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.emb = emb

        # for LTM
        self._tlwh = bbox.float()
        self.sequence = self._tlwh.unsqueeze(0)
        self.d_sequence = torch.zeros((1, 1, 4))
        self.pred = None  # Tensor(1, 4)

        # for PBA
        bbox_tlbr = bbox[0].numpy().copy()
        bbox_tlbr[2:] += bbox_tlbr[:2]
        self.trk_seq = [bbox_tlbr]
        self.last_observation = np.array([-1, -1, -1, -1, -1])

    def update(self, bbox):
        if bbox is not None:
            self.last_observation = bbox
            self.trk_seq[-1] = bbox[:4]

            if self.time_since_update > 1:
                self.oos()

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            # motion update
            new_tlwh = self.tlbr_to_tlwh(bbox[:4])
            self.sequence = torch.cat((self.sequence, new_tlwh.unsqueeze(1)), dim=1)
            self.d_sequence = torch.cat((self.d_sequence, self.sequence[:, -1:, :] - self.sequence[:, -2:-1, :]), dim=1)
        else:
            self.sequence = torch.cat((self.sequence, self.pred.unsqueeze(1)), dim=1)
            self.d_sequence = torch.cat((self.d_sequence, self.sequence[:, -1:, :] - self.sequence[:, -2:-1, :]), dim=1)

    @staticmethod
    def multi_predict(stracks, with_scene=False, cmc_affine=None):
        """

        :param stracks:
        :param with_scene:
        :param cmc_affine: ndarray(frame_count, 6)
        :return:
        """
        if len(stracks) > 0:
            max_seq_len = 0
            for st in stracks:
                if max_seq_len < st.sequence.shape[1]:
                    max_seq_len = st.sequence.shape[1]
            multi_sequence = []
            multi_d_sequence = []
            for st in stracks:
                if st.sequence.shape[1] < max_seq_len:
                    pad_dim = max_seq_len - st.sequence.shape[1]
                    multi_sequence.append(
                        F.pad(st.sequence.permute(0, 2, 1), pad=(pad_dim, 0, 0, 0), mode='constant', value=0).permute(
                            0, 2, 1))
                    multi_d_sequence.append(
                        F.pad(st.d_sequence.permute(0, 2, 1), pad=(pad_dim, 0, 0, 0), mode='constant', value=0).permute(
                            0, 2, 1))
                else:
                    multi_sequence.append(st.sequence)
                    multi_d_sequence.append(st.d_sequence)
            multi_sequence = torch.cat(multi_sequence, dim=0).permute(1, 0, 2).to(
                torch.float32)
            multi_d_sequence = torch.cat(multi_d_sequence, dim=0).permute(1, 0, 2).to(
                torch.float32)
            multi_pred = LongTermMotionTracker.motion_model(multi_sequence.cuda(),
                                                            multi_d_sequence.cuda(),)  # (track_num, 4)

            multi_pred_abs = multi_pred.abs.detach().cpu()[0][0]
            for i in range(multi_pred_abs.shape[0]):
                if stracks[i].time_since_update > 0:
                    stracks[i].hit_streak = 0
                stracks[i].time_since_update += 1
                stracks[i].pred = multi_pred_abs[i:i + 1, :]
                pred_tlbr = multi_pred_abs[i, :].numpy().copy()
                pred_tlbr[2:] = pred_tlbr[:2] + pred_tlbr[2:]
                stracks[i].trk_seq.append(pred_tlbr)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

    def oos(self):
        last_bbox = self.trk_seq[-1]
        start_bbox = self.trk_seq[-(self.time_since_update + 1)]
        time_gap = self.time_since_update
        d_tlbr = (last_bbox - start_bbox) / time_gap
        for i in range(self.time_since_update - 1):
            revised_bbox = start_bbox + (i + 1) * d_tlbr
            self.trk_seq[-self.time_since_update + i] = revised_bbox.copy()
            revised_bbox[2:] -= revised_bbox[:2]
            self.sequence[0][-self.time_since_update + i] = torch.from_numpy(revised_bbox)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """
            tlbr: np.array, (4,)
        """
        ret = torch.from_numpy(tlbr.copy()).unsqueeze(0)
        ret[:, 2:] -= ret[:, :2]
        return ret


class LongTermMotionTracker:
    motion_model_cfg = None
    motion_epoch = None

    assa_model_cfg = None
    assa_epoch = None

    def __init__(
            self,
            det_thresh,
            max_lost_age=30,
            min_hits=3,
            iou_threshold=0.3,
            w_association_emb=0.75,
            alpha_fixed_emb=0.95,
            aw_param=0.5,
            embedding_off=False,
            **kwargs,
    ):
        self.max_lost_age = max_lost_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        STrack.count = 0

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset)
        self.cmc = CMCComputer()
        self.cmc_affine_list = []
        self.embedding_off = embedding_off

        self.pba = kwargs["args"].PBA
        self.ztrm = kwargs["args"].ZTRM
        self.ztrm_emb_thr = kwargs["args"].ztrm_emb_thr
        self.ztrm_iou_thr = kwargs["args"].ztrm_iou_thr
        self.max_zombie_age = kwargs["args"].max_zombie_age
        self.zombie_trackers = []

    @classmethod
    def load_motion_model(cls):
        cls.motion_model = get_motion_model(
            motion_model_cfg=cls.motion_model_cfg,
            motion_epoch=cls.motion_epoch
        )

    @classmethod
    def load_assa_model(cls):
        cls.assa_model = get_assa_model(cls.assa_model_cfg, cls.assa_epoch)

    def update(self, output_results, img_tensor, img_numpy, tag):
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        bboxes[:, :4] /= scale

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        dets = dets[dets[:, -1] > 0.1]
        scores = scores[scores > 0.1]
        det_idx = np.asarray(list(range(len(dets))))
        det_idx_high = det_idx[scores > self.det_thresh]
        det_idx_low = det_idx[np.logical_and(scores > 0.1, scores < self.det_thresh)]

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        # CMC
        transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
        self.cmc_affine_list.append(transform.reshape(6, ))
        for trk in self.trackers:
            trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        matched, unmatched_dets, unmatched_trks = self._match(dets, dets_embs, img_numpy, det_idx_high, det_idx_low)
        rm_det_idx_idx = oai(dets, unmatched_dets, self.trackers)
        # remove idx in rm_det_idx from unmatched_detections
        unmatched_dets = np.delete(unmatched_dets, rm_det_idx_idx)

        ret = []
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = STrack(
                STrack.tlbr_to_tlwh(dets[i, :4]), emb=dets_embs[i]
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.sequence[0, -1].numpy().copy()
                d[2:] += d[:2]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_lost_age:
                last_bbox = trk.last_observation[:4]
                if (last_bbox[:2] > 0).all() and last_bbox[2] < img_numpy.shape[1] and last_bbox[3] < img_numpy.shape[
                    0]:
                    self.zombie_trackers.append(self.trackers[i])
                self.trackers.pop(i)

        for j, ztrk in enumerate(self.zombie_trackers):
            if ztrk.time_since_update > self.max_zombie_age:
                self.zombie_trackers.pop(j)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _match(self, dets, dets_embs, img_numpy=None, det_idx_high=None, det_idx_low=None):
        # get predicted locations from existing trackers.
        if self.ztrm:
            stracks_pool = joint_stracks(self.trackers, self.zombie_trackers)
        else:
            stracks_pool = self.trackers
        STrack.multi_predict(stracks_pool, cmc_affine=np.asarray(self.cmc_affine_list))
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        for t, trk in enumerate(trks):
            if self.trackers[t].last_observation.sum() < 0:
                pos = self.trackers[t].sequence[0, -1].numpy().copy()
            else:
                pos = self.trackers[t].pred[0].numpy().copy()
            pos[2:] += pos[:2]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)
        last_boxes = np.array([trk.last_observation for trk in self.trackers])

        """
            First round of association
        """
        if self.pba:
            trk_sequences = [np.array(trk.trk_seq) for trk in self.trackers]
            matched, unmatched_dets, unmatched_trks = associate_with_inter_cost(
                dets[det_idx_high],
                trks,
                dets_embs[det_idx_high],
                trk_embs,
                self.iou_threshold,
                trk_sequences,
                self.assa_model,
                self.w_association_emb,
                self.aw_param,
                self.embedding_off,
            )
        else:
            matched, unmatched_dets, unmatched_trks = associate_with_iou(
                dets[det_idx_high],
                trks,
                dets_embs[det_idx_high],
                trk_embs,
                self.iou_threshold,
                self.w_association_emb,
                self.aw_param,
                self.embedding_off,
            )
        matched, unmatched_dets, unmatched_trks = calibrate_idx(matched, unmatched_dets, unmatched_trks,
                                                                det_idx_high, np.asarray(list(range(len(trks)))))
        """
            Second round of associaton
        """
        if (len(unmatched_dets) + len(det_idx_low)) > 0 and len(unmatched_trks) > 0:
            det_idx_second = np.concatenate((unmatched_dets, det_idx_low), axis=0)
            dets_second = dets[det_idx_second]
            left_trks = last_boxes[unmatched_trks]

            iou_left = iou_batch(dets_second, left_trks)
            iou_left = np.array(iou_left)

            if self.pba:
                max_seq_len = max([len(trk) for trk in trk_sequences])
                for i in range(len(trk_sequences)):
                    if len(trk_sequences[i]) < max_seq_len:
                        trk_sequences[i] = np.pad(trk_sequences[i], ((max_seq_len - len(trk_sequences[i]), 0), (0, 0)),
                                                  'constant',
                                                  constant_values=(0, 0))

                trk_seq = np.asarray(trk_sequences)[unmatched_trks][:, :-1, :]
                det_rel_pos = relative_position(dets_second[:, :4])
                trk_rel_pos = relative_position(left_trks[:, :4])
                pos_simi = self.assa_model(
                    torch.from_numpy(trk_seq).unsqueeze(0).to(torch.float32).cuda(),
                    torch.from_numpy(trk_rel_pos).unsqueeze(0).to(torch.float32).cuda(),
                    torch.from_numpy(dets_second[:, :4]).unsqueeze(0).to(torch.float32).cuda(),
                    torch.from_numpy(det_rel_pos).unsqueeze(0).to(torch.float32).cuda())
                final_cost = - (pos_simi.view(pos_simi.shape[1], pos_simi.shape[2]).detach().t().cpu().numpy())
            else:
                final_cost = -iou_left

            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(final_cost)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = det_idx_second[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    matched = np.concatenate((matched.reshape(-1, 2), np.asarray([[det_ind, trk_ind]])), axis=0)
                    to_remove_trk_indices.append(trk_ind)
                    if det_ind not in det_idx_low:  # only high score detections can be initialized as new tracks
                        to_remove_det_indices.append(det_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if not self.ztrm:
            return matched, unmatched_dets, unmatched_trks
        """
            Zombie Track Re-Match
        """
        mca_dets_indices = []
        for i in unmatched_dets:
            if (dets[i][:2] > 0).all() and dets[i][2] < img_numpy.shape[1] and dets[i][3] < img_numpy.shape[0]:
                mca_dets_indices.append(i)
        mca_dets_indices = np.array(mca_dets_indices)
        left_dets = dets[mca_dets_indices] if mca_dets_indices.shape[0] > 0 else []  # MCA-v2
        left_dets_embs = dets_embs[mca_dets_indices] if mca_dets_indices.shape[0] > 0 else []

        ztrk_pos = np.zeros((len(self.zombie_trackers), 5))  # MCA-v2
        ztrk_embs = []
        for ztrk in self.zombie_trackers:
            pos = ztrk.pred[0].numpy().copy()
            pos[2:] += pos[:2]
            ztrk_pos[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            ztrk_embs.append(ztrk.get_emb())
        ztrk_embs = np.array(ztrk_embs)

        mca_matched, mca_unmatched_dets, mca_unmatched_trks = associate_zombie_tracks(
            left_dets_embs,
            ztrk_embs,
            left_dets,
            ztrk_pos,
            emb_threshold=self.ztrm_emb_thr,
            iou_threshold=self.ztrm_iou_thr,
        )
        to_remove_det_indices = []
        to_remove_ztrk_indices = []
        for m in mca_matched:
            det_ind, ztrk_ind = mca_dets_indices[m[0]], m[1]
            to_remove_det_indices.append(det_ind)
            to_remove_ztrk_indices.append(ztrk_ind)
            ztrk = self.zombie_trackers[ztrk_ind]
            matched = np.concatenate((matched.reshape(-1, 2), np.asarray([[det_ind, len(self.trackers)]])), axis=0)  # MCA-v2
            self.trackers.append(ztrk)
        for m in mca_unmatched_trks:
            self.zombie_trackers[m].update(None)
        for i in sorted(to_remove_ztrk_indices, reverse=True):
            self.zombie_trackers.pop(i)

        unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))

        return matched, unmatched_dets, unmatched_trks

    def dump_cache(self):
        self.embedder.dump_cache()


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res
