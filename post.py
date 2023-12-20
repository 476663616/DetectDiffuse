import argparse
import os

# import mmcv
import numpy as np
import pandas as pd
from pycocotools.mask import iou as IOU
import torch

from test import prepocess

# from mmcv.image import tensor2imgs
# from mmdet.datasets import build_dataloader, build_dataset

score_thr = 0.1
iou_thr = 0.01    


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--post_name', default='test_post.pth', help='post_name')
    parser.add_argument('--outfile', default='./post_result/0711R50_4Step_uniform_buffer_500box',
                        help='output result file in pickle format')
    parser.add_argument('--show_thr', type=float, default=0.5, help='show results')#0.95
    parser.add_argument('--show_num', type=int, default=1)
    args = parser.parse_args()
    return args


def show_result(data, result, dataset=None, outfile=None, score_thr=score_thr):
    bbox_result = result
    if isinstance(data['img'][0], list):
        img_tensor = data['img'][0][2]
    else:
        img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]  ###############################meta----->metas
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        bboxes = np.vstack(bbox_result)
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        if outfile:
            if not isinstance(img_meta['filename'], list):
                outfile = os.path.join(outfile, img_meta['filename'].split('/')[-1])
            else:
                outfile = os.path.join(outfile, img_meta['filename'][2].split('/')[-1])
        mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            score_thr=score_thr,
            show=False,#True
            out_file=outfile)


def single_show(data_loader, outputs, outfile=None, show_thr=0.3):
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        #debug
        # if i==63:
        #     print(data['img_metas'][0].data[0][0]['ori_filename'])
        show_result(data=data, result=outputs[i], outfile=outfile, score_thr=show_thr)

        if isinstance(data['img'][0], list):
            batch_size = data['img'][0][1].size(0)
        else:
            batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


def xyxy2xywh(bbox):
    _bboxs = [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
    return _bboxs


def xywh2xyxy(bbox):
    _bboxs = []
    for _bbox in bbox:
        _bboxs.append([_bbox[0], _bbox[1], _bbox[2] + _bbox[0] - 1, _bbox[3] + _bbox[1] - 1])
    return _bboxs


def single_post(dets):
    new_dets = []
    for i, det in enumerate(dets):
        new_det = []
        flag = dict()
        flag_set = set()

        g, det = xyxy2xywh(det[0].tolist())
        if len(g) > 1:
            iscrowd = [0 for _ in range(len(g))]
            ious = IOU(g, g, iscrowd)                  

            if sum(sum(ious == 0)) != ious.shape[0] * ious.shape[1] - ious.shape[0]:
                for j in range(ious.shape[0]):
                    for k in range(j+1, ious.shape[1]):
                        if ious[j][k] > iou_thr:
                            if j not in flag_set and k not in flag_set:
                                if j in flag:                                
                                    flag[j].append(k)                       
                                else:                                       
                                    flag[j] = [j, k]
                                flag_set.add(j)
                                flag_set.add(k)
                            else:
                                for key in flag:
                                    if j in flag[key]:
                                        flag[key].append(k)
                                        flag_set.add(k)
                                    elif k in flag[key]:
                                        flag[key].append(j)
                                        flag_set.add(j)

                for j in range(ious.shape[0]):          
                    if j in flag_set:
                        if j in flag:
                            overlap = list(set(flag[j]))
                            det_temp = [det[0][index] for index in overlap]
                            det_temp = np.array(det_temp)
                            # det_temp = [min(det_temp[:, 0]), min(det_temp[:, 1]),
                            #             max(det_temp[:, 2]), max(det_temp[:, 3]), max(det_temp[:, 4])]
                            temp_index = np.argwhere(det_temp == max(det_temp[:, 4]))[0][0]
                            det_temp = det_temp[temp_index]
                            new_det.append(det_temp)
                    else:
                        new_det.append(det[0][j].tolist())
                new_det = [np.array(new_det)]
        if new_det:
            new_dets.append(new_det)
        elif det[0].shape[0]:
            new_dets.append(det)
        else:
            new_dets.append([np.array([[], [], [], [], []]).T])
    return new_dets


def tem_post(dets):
    new_dets = []
    for i in range(0, len(dets), num_frame):
        flag_pos = 0
        base = []
        for j in range(i, i+num_frame):
            for det in dets[j][0]:
                pos = is_same_tem(base, det, j)
                if pos == -1:
                    base.append({'det': det.tolist(), 'img_id': j, 'pos_id': flag_pos, 'flag': 1})
                    flag_pos += 1
                else:
                    base.append({'det': det.tolist(), 'img_id': j, 'pos_id': pos, 'flag': 2})

        pos_head = list(filter(lambda x: x['flag'] == 1, base))
        pos_end = list(filter(lambda x: x['flag'] == 2, base))

        for head in pos_head:
            for end in pos_end:
                if head['img_id'] != end['img_id']:
                    if compute_iou(head['det'], end['det']) > iou_thr:
                        for base_index in range(len(base)):
                            if base[base_index]['pos_id'] == head['pos_id']:
                                base[base_index]['pos_id'] = end['pos_id']

        new_base = []
        for pos in range(flag_pos):
            x = list(filter(lambda x: x['pos_id'] == pos, base))
            if len(x) >= frame_thr:
                new_base += x

        for j in range(i, i+num_frame):
            new_det = []
            x = list(filter(lambda x: x['img_id'] == j, new_base))
            for xx in x:
                new_det.append(xx['det'])
            if not new_det:
                new_det = [np.array([[], [], [], [], []]).T]
            else:
                new_det = [np.array(new_det)]
            new_dets.append(new_det)
    return new_dets


def is_same_tem(base, det, img_id):
    pos = -1
    for base_det in base:
        if base_det['img_id'] == img_id - 1:
            if compute_iou(base_det['det'], det) > iou_thr:
                pos = base_det['pos_id']
                if base_det['flag'] == 2:
                    base_det['flag'] = 0
                break
    return pos


def multi_post(dets):
    new_dets = []
    for i in range(0, len(dets), num_frame):
        flag_pos = 0
        base = []
        for j in range(i, i+num_frame):
            if j == 12:
                print()
            if dets[j] == []:
                if j == i:
                    dets[j] = dets[j+1]
                else:
                    dets[j] = dets[j-1]
            if dets[j] == []:
                continue
            for det in dets[j]:
                pos = is_same_pos(base, det[0], j)
                if pos == -1:
                    base.append({'det': det[0].tolist(), 'img_id': j, 'pos_id': flag_pos})
                    flag_pos += 1
                else:
                    base.append({'det': det[0].tolist(), 'img_id': j, 'pos_id': pos})

        new_base = []
        for pos in range(flag_pos):
            x = list(filter(lambda x: x['pos_id'] == pos, base))
            if len(x) >= frame_thr:
                new_base += x

        for j in range(i, i+num_frame):
            new_det = []
            x = list(filter(lambda x: x['img_id'] == j, new_base))
            for xx in x:
                new_det.append(xx['det'])
            if not new_det:
                new_det = [np.array([[], [], [], [], []]).T]
            else:
                new_det = [np.array(new_det)]
            new_dets.append(new_det)
    return new_dets


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def similarity(rec1, rec2, i, j):
    pass


def is_same_pos(base, det, img_id):
    pos = -1
    for base_det in base:
        if base_det['img_id'] != img_id:
            if compute_iou(base_det['det'], det) > iou_thr:
                pos = base_det['pos_id']
                break
    return pos


def find_num(img_ids):
    out = []
    for i in range(num_frame):
        if i not in img_ids:
            out.append(i)
    return out

#TODO 后处理通过半径分析来优化，定位狭窄中心
def chazhi(dets):
    outputs = []
    for i in range(0, len(dets), num_frame):
        flag_pos = 0
        base = []
        for j in range(i, i+num_frame):
            for det in dets[j][0]:
                pos = is_same_pos(base, det, j)
                if pos == -1:
                    base.append({'det': det.tolist(), 'img_id': j, 'pos_id': flag_pos})
                    flag_pos += 1
                else:
                    base.append({'det': det.tolist(), 'img_id': j, 'pos_id': pos})

        new_base = []
        for pos in range(flag_pos):
            x = list(filter(lambda x: x['pos_id'] == pos, base))
            img_ids = [box['img_id'] % 9 for box in x]
            num_id = i // 9
            no_ids = find_num(img_ids)
            x1, y1, x2, y2, score = [], [], [], [], []
            for j in range(i, i+num_frame):
                xx = list(filter(lambda x: x['img_id'] == j, x))
                if len(xx) != 0:
                    x1.append(xx[0]['det'][0])
                    y1.append(xx[0]['det'][1])
                    x2.append(xx[0]['det'][2])
                    y2.append(xx[0]['det'][3])
                    score.append(xx[0]['det'][4])
                else:
                    x1.append(np.nan)
                    y1.append(np.nan)
                    x2.append(np.nan)
                    y2.append(np.nan)
            x1 = pd.DataFrame(x1).interpolate().bfill().values
            y1 = pd.DataFrame(y1).interpolate().bfill().values
            x2 = pd.DataFrame(x2).interpolate().bfill().values
            y2 = pd.DataFrame(y2).interpolate().bfill().values
            for no_id in no_ids:
                # x.append({'det': [x1[no_id-1][0], y1[no_id-1][0], x2[no_id-1][0], y2[no_id-1][0], max(score)],
                #           'img_id': no_id + num_id * 9, 'pos_id': pos})
                x.append({'det': [x1[no_id][0], y1[no_id][0], x2[no_id][0], y2[no_id][0], max(score)],
                          'img_id': no_id + num_id * 9, 'pos_id': pos})
            new_base += x

        # list to numpy
        for j in range(i, i+num_frame):
            new_det = []
            x = list(filter(lambda x: x['img_id'] == j, new_base))
            for xx in x:
                new_det.append(xx['det'])
            if not new_det:
                new_det = [np.array([[], [], [], [], []]).T]
            else:
                new_det = [np.array(new_det)]
            outputs.append(new_det)
    return outputs

def get_bbox(anns_list, thr):
    bboxes = []
    # for ann in anns_list:
    if anns_list['score'] >= thr:
        bbox = xywh2xyxy([anns_list['bbox']])
        bbox = bbox[0]
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3], anns_list['score']]
        bboxes.append(bbox)
    return bboxes


def pth2pkl(all_pre):
    dets_all = []
    for pre in all_pre:
        dets = []
        pred = pre['instances']
        for i in range(len(pred)):
            det = pred[i]
            stenosis_bbox = get_bbox(det, 0.5)
            if stenosis_bbox != []:
                stenosis_bbox = np.stack(stenosis_bbox, axis=0)
                dets.append(stenosis_bbox)
            else:
                for thr_new in range(int(0.5*10), 0, -1):
                    print('the thr now is: ', thr_new/10)
                    stenosis_bbox = get_bbox(det, thr_new/10)
                    if stenosis_bbox != []:
                        stenosis_bbox = np.stack(stenosis_bbox, axis=0)
                        dets.append(stenosis_bbox)
                        break
        dets_all.append(dets)
    return dets_all

def det2pth(dets, save_root):
    category_id = 0
    all_result = []
    for i in range(len(dets)):
        frame_result = dict()
        frame_result['image_id'] = i 
        frame_result['instances'] = []
        for det in dets[i][0]:
            instance = dict()
            instance['image_id'] = i
            instance['category_id'] = 0
            instance['bbox'] = [det[0], det[1], det[2], det[3]]
            instance['bbox'] = xyxy2xywh(instance['bbox'])
            instance['score'] = det[-1]
            frame_result['instances'].append(instance)
        all_result.append(frame_result)
    torch.save(all_result, save_root)

    
if __name__ == '__main__':

    pred_path = './output/0711R50_4Step_uniform_buffer_500box/inference/instances_predictions.pth'
    # pred_path = './instances_predictions.pth'
    data_root = './datasets/coco/train_2017/'
    gt_root = './datasets/coco/annotations/50_test_1.json'
    save_root = './post_result/0711R50_4Step_uniform_buffer_500box/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    num_frame = 9
    frame_thr = 5
    args = parse_args()
    # cfg = mmcv.Config.fromfile(args.config)
    # cfg.data.test.test_mode = True

    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=False,
    #     shuffle=False)
    result = torch.load(pred_path)
    new_result = pth2pkl(result)
    # result = prepocess(result)
    # dets = mmcv.load(args.out)
    new_dets_single = []
    # new_dets_single = single_post(dets)                 
    # new_dets_single = single_post(new_dets_single)      
    if not new_dets_single:
        new_dets_single = new_result
    new_dets = multi_post(new_dets_single)

    chazhi_dets = chazhi(new_dets)

    # tem_dets = tem_post(dets)

    outfile = os.path.join(args.outfile, str(0))
    outfile1 = os.path.join(args.outfile, str(1))
    outfile2 = os.path.join(args.outfile, str(2))
    outfile3 = os.path.join(args.outfile, str(3))
    outfile4 = os.path.join(args.outfile, str(4))
    # if not os.path.exists(outfile):
    #     os.makedirs(outfile)
    # if not os.path.exists(outfile1):
    #     os.makedirs(outfile1)
    # if not os.path.exists(outfile2):
    #     os.makedirs(outfile2)
    # if not os.path.exists(outfile3):
    #     os.makedirs(outfile3)
    # if not os.path.exists(outfile4):
    #     os.makedirs(outfile4)

    # single_show(data_loader, dets, outfile, args.show_thr)
    # single_show(data_loader, new_dets_single, outfile1, args.show_thr)
    # single_show(data_loader, tem_dets, outfile2, args.show_thr)
    # single_show(data_loader, tem_dets, outfile3, args.show_thr)
    # single_show(data_loader, chazhi_dets, outfile4, args.show_thr)

    # outfile11 = args.outfile + '_single.pkl'
    # mmcv.dump(new_dets_single, outfile11)

    # outfile22 = args.outfile + '_multi.pkl'
    # mmcv.dump(new_dets, outfile22)

    # outfile33 = args.outfile + '_tem.pkl'
    # mmcv.dump(tem_dets, outfile33)

    outfile44 = args.outfile + '/'+args.post_name
    det2pth(chazhi_dets, outfile44)
    # mmcv.dump(chazhi_dets, outfile44)

    # outfile44 = args.outfile
    # mmcv.dump(chazhi_dets, outfile44)