import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from . import BaseActor
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ...utils.heapmap_utils import generate_heatmap


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 2
        assert len(data['search_images']) == 2

        out_dict_list = []
        forward_data_1 = data['template_images'][0:1, :, :, :, :]
        forward_data_2 = data['search_images'][0:1, :, :, :, :]
        forward_data_3 = data['template_label'][0:1, :]
        forward_data_4 = data['search_label'][0:1, :]
        forward_data_5 = data['template_anno'][0:1, :, :]
        backward_data_1 = data['template_images'][1:2, :, :, :, :]
        backward_data_2 = data['search_images'][1:2, :, :, :, :]
        backward_data_3 = data['template_label'][1:2, :]
        backward_data_4 = data['search_label'][1:2, :]
        backward_data_5 = data['template_anno'][1:2, :, :]
        for stage in range(2):
            # rewrite
            if stage == 0:
                data['template_images'] = forward_data_1
                data['search_images'] = forward_data_2
                data['template_label'] = forward_data_3
                data['search_label'] = forward_data_4
                data['template_anno'] = forward_data_5
            else:
                data['template_images'] = backward_data_1
                data['search_images'] = backward_data_2
                data['template_label'] = backward_data_3
                data['search_label'] = backward_data_4
                data['template_anno'] = backward_data_5

            template_list = []
            for i in range(self.settings.num_template):
                template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
                # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])
                template_list.append(template_img_i)

            for i in range(self.settings.num_search):
                print(i)

            search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])
            # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])
            # NOTE: 传入模态标签
            search_label = data['search_label'].view(-1, 1)  # (1, B) -> (B, 1)
            template_label = data['template_label'].view(-1, 1)
            if not torch.all(torch.logical_or(search_label == 0, search_label == 1)):
                print(search_label)
                raise ValueError("search_label 中包含错误的标签！")
            if not torch.all(torch.logical_or(template_label == 0, template_label == 1)):
                print(template_label)
                raise ValueError("template_label 中包含错误的标签！")

            box_mask_z = None
            ce_keep_rate = None
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                                data['template_anno'][0])

                ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
                ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
                ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

            if len(template_list) == 1:
                template_list = template_list[0]

            out_dict = self.net(template=template_list,
                                search=search_img,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                template_label=template_label,
                                search_label=search_label,
                                infer=False)
            out_dict_list.append(out_dict)

        return out_dict_list

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  # 定义 0 tensor，并指定GPU设备
        total_status = {}

        forward_gt = gt_dict['search_anno'][0:1, :, :]
        backward_gt = gt_dict['search_anno'][1:2, :, :]

        for i in range(len(pred_dict)):
            if i == 0:
                gt_dict['search_anno'] = forward_gt
            else:
                gt_dict['search_anno'] = backward_gt
            # gt gaussian map
            gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                                self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            # weighted sum
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
                'focal'] * location_loss
            total_loss += loss

            if return_status:
                mean_iou = iou.detach().mean()
                status = {f"stage-{i}_Loss/total": loss.item(),
                          f"stage-{i}_Loss/giou": giou_loss.item(),
                          f"stage-{i}_Loss/l1": l1_loss.item(),
                          f"stage-{i}_Loss/location": location_loss.item(),
                          f"stage-{i}_IoU": mean_iou.item()}

                total_status.update(status)

        consist_loss = self.objective['consist'](pred_dict[0]['backbone_feat'], pred_dict[1]['backbone_feat'])
        consist_loss += self.objective['consist'](pred_dict[0]['pred_boxes'], pred_dict[1]['pred_boxes'])
        consist_loss += self.objective['consist'](pred_dict[0]['score_map'], pred_dict[1]['score_map'])
        consist_loss += self.objective['consist'](pred_dict[0]['size_map'], pred_dict[1]['size_map'])
        consist_loss += self.objective['consist'](pred_dict[0]['offset_map'], pred_dict[1]['offset_map'])
        total_loss += consist_loss

        if return_status:
            # status for log
            status = {"Loss/total": total_loss.item(),
                      "Loss/consist": consist_loss.item(), }
            total_status.update(status)
            return total_loss, total_status
        else:
            return total_loss
