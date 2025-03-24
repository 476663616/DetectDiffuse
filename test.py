import torch
import os
from pycocotools.coco import COCO
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
import cv2
import numpy as np
import xlwt
import copy

def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 j 行，第 i 列
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(j, i, data[j])
        i = i + 1

    f.save(file_path)

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

def is_same_pos(base, det, img_id):
    pos = -1
    for base_det in base:
        if base_det['img_id'] != img_id:#滤除同一张图像上的框
            iou=compute_iou(base_det['det'], det)
            if iou>0.01:
            # if iou:
                pos = base_det['pos_id']
                break
    return pos

def load_annotations(ann_file):
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    gt_bboxs = []
    for imgid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=imgid)
        anns = coco.loadAnns(ann_ids)
        gt_bbox = [np.array([a['bbox'] for a in anns])]
        gt_bboxs.append(gt_bbox)
    return gt_bboxs

def nearest(base, det, img_id):
    pos = -1
    min_distance = float('inf')
    for base_det in base:
        if base_det['img_id'] == img_id - 1:
            x1 = (base_det['det'][0] + base_det['det'][2]) / 2
            y1 = (base_det['det'][1] + base_det['det'][3]) / 2
            x2 = (det[0] + det[2]) / 2
            y2 = (det[1] + det[3]) / 2
            distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)#计算中心坐标距离
            if distance < min_distance:
                min_distance = distance
                pos = base_det['pos_id']
    return pos

def xywh2xyxy(bbox):
    _bboxs = []
    for _bbox in bbox:
        _bboxs.append([_bbox[0], _bbox[1], _bbox[2] + _bbox[0] - 1, _bbox[3] + _bbox[1] - 1])
    return _bboxs

def eval_frame(dets, gt_bboxs,final_eva):
    TP = 0
    FP = 0
    FN = 0

    gt_eva=[]
    pre_eva=[]
    TP_eva=[]
    FP_eva=[]
    FN_eva=[]
    
    for i in range(0, len(dets)):#range(0, 279, 9)
        flag_pos = 0
        base_dets = []
        base_gt_bboxs = []
        det_pos = set()
        gt_pos = set()
        
            
        for det in dets[i]['instances']:# box index
            pos = is_same_pos(base_dets, det['bbox'], i)#根据重叠程度返回框的id
            if pos == -1:#如果在前面的帧中没有匹配的box
                base_dets.append(
                    {'det': det['bbox'], 'img_id': i, 'pos_id': flag_pos})
                det_pos.add(flag_pos)#新增框的id
                flag_pos += 1
            else:
                base_dets.append(
                    {'det': det['bbox'], 'img_id': i, 'pos_id': pos})

        for k, gt_bbox in enumerate(gt_bboxs[i][0]):
            base_gt_bboxs.append({'det': gt_bbox.tolist(), 'img_id': i, 'pos_id': k})
            gt_pos.add(k)#新增框的id
        


        for img_index in range(base_gt_bboxs[0]['img_id'], base_gt_bboxs[-1]['img_id'] + 1):#金标准序列帧的框id（每帧一样）#############img_id
            if img_index == 12:
                print()
            TP_temp = 0
            FP_temp = 0
            FN_temp = 0 
            gt_temp = set()
            det_temp = set()
            #gt和pred匹配标志
            gt_pos_index_temp = -1
            det_pos_index_temp = -1
            gt_x = [xx['det'] for xx in filter(lambda x: x['img_id'] == img_index, base_gt_bboxs)]#一个序列中同一个id对应的框
            gt_x = xywh2xyxy(gt_x)
            det_x = [xx['det'] for xx in filter(lambda x: x['img_id'] == img_index, base_dets)]
            det_x = xywh2xyxy(det_x)
            for temp in range(len(gt_x)):
                gt_temp.add(temp)
            for temp in range(len(det_x)):
                det_temp.add(temp)
            gt_eva.append(len(gt_temp))
            pre_eva.append(len(det_temp))
            for det_xx in det_x:
                for gt_xx in gt_x:
                    # print(compute_iou(gt_xx, det_xx))
                    flag_index = 0
                    if compute_iou(gt_xx, det_xx) >= 0.01:
                        flag_index = 1
                        gt_pos_index_temp = gt_x.index(gt_xx)
                        det_pos_index_temp = det_x.index(det_xx)
                        if flag_index == 1:
                            TP += 1
                            TP_temp+=1
                            if det_pos_index_temp in det_temp:
                                det_temp.remove(det_pos_index_temp)
                            if gt_pos_index_temp in gt_temp:
                                gt_temp.remove(gt_pos_index_temp)
                            else:
                                continue
    #TODO: 返回每个序列的TP FP FN，加入到list或字典
            FP_temp=len(list(det_temp))
            FN_temp=len(list(gt_temp))

            TP_eva.append(TP_temp)
            FP_eva.append(FP_temp)
            FN_eva.append(FN_temp)
            
            #根据探测框数目来计算
            FN += FN_temp#加上每个序列的FN框数目
            FP += FP_temp#加上每个序列的FP框数目
    final_eva.append(gt_eva)
    final_eva.append(pre_eva)
    final_eva.append(TP_eva)
    final_eva.append(FP_eva)
    final_eva.append(FN_eva)

    return TP, FN, FP,final_eva

def is_xywh(det):
    h = det[3] - det [1]
    w = det[2] - det[0]
    if h<0 or w<0:
        return True
    else:
        return False

def show_img(results, gts, save_root):
    image_root = './datasets/8bit/testset/image/'
    for result in results:
        img_id = result['image_id']
        image_name = gts.loadImgs(img_id)[0]['file_name']
        img = read_image(image_root+image_name,format='BGR')
        if result['instances'] == []:
            continue
        else:      
            image_dicts = result['instances']
            # if img_id == 19:
            #     print()
            for image_dict in image_dicts:
                if not is_xywh(image_dict['bbox']):
                    image_dict['bbox'] = [image_dict['bbox'][0], image_dict['bbox'][1], image_dict['bbox'][2]-image_dict['bbox'][0], image_dict['bbox'][3]-image_dict['bbox'][1]]
                if image_dict['category_id'] == 1:
                    continue
                visualizer = Visualizer(img, metadata={'thing_classes':['stenosis']})
                img_visual = visualizer.draw_stenosis_dict(image_dict)
                img = img_visual.get_image()
            cv2.imwrite(save_root+image_name, img)

def  show_img_contrast(results, gts, save_root):
    image_root = '/home/xinyul/python_exercises/data/lesions/COVID_coco/images/'
    for result in results:
        img_id = result['image_id']
        image_name = gts.loadImgs(img_id)[0]['file_name']
        anns_id_list = gts.getAnnIds(img_id, 1)
        anns_list = gts.loadAnns(anns_id_list)
        img = read_image(image_root+image_name,format='BGR')
        if result['instances'] == []:
            continue
        else:      
            image_dicts = result['instances']
            # if img_id == 19:
            #     print()
            for image_dict in image_dicts:
                if not is_xywh(image_dict['bbox']):
                    image_dict['bbox'] = [image_dict['bbox'][0], image_dict['bbox'][1], image_dict['bbox'][2]-image_dict['bbox'][0], image_dict['bbox'][3]-image_dict['bbox'][1]]
                if image_dict['category_id'] == 1:
                    continue
                visualizer = Visualizer(img, metadata={'thing_classes':['stenosis']})
                img_visual = visualizer.draw_stenosis_dict(image_dict)
                img = img_visual.get_image()
            for ann in anns_list:
                gt_dict = {}
                gt_dict['bbox'] = ann['bbox']
                gt_dict['category_id'] = 0
                gt_dict['score'] = 1
                gt_dict['image_id'] = img_id
                visualizer = Visualizer(img, metadata={'thing_classes':['stenosis']})
                img_visual = visualizer.draw_gt_dict(gt_dict)
                img = img_visual.get_image()

            cv2.imwrite(save_root+image_name, img)

def prepocess(results, thr=0.35):
    new_results = []
    for result in results:
        temp_result = {}
        temp_result['image_id'] = result['image_id']
        dets = result['instances']
        tmp_dets = []
        for det in dets:
            if det['score'] >= thr:
                tmp_dets.append(det)
        temp_result['instances'] = tmp_dets
        new_results.append(temp_result)
    return new_results

def gt_vis(gt, save_root):
    from utils import maybe_make_dir
    maybe_make_dir(save_root)
    gt_root = './datasets/8bit/testset/image/'
    for i in range(len(gt.imgs)):
        img = gt.imgs[i]
        img_id = img['id']
        image_name = img['file_name']
        anns_id_list = gt.getAnnIds(img_id, 1)
        anns_list = gt.loadAnns(anns_id_list)
        image_dict = {}
        img_arr = read_image(gt_root+image_name,format='BGR')
        for ann in anns_list:
            image_dict['image_id'] = img_id
            image_dict['category_id'] = 0
            image_dict['bbox'] = ann['bbox']
            image_dict['score'] = 1.0
            visualizer = Visualizer(img_arr, metadata={'thing_classes':['stenosis']})
            img_visual = visualizer.draw_stenosis_dict(image_dict)
            img_arr = img_visual.get_image()
        cv2.imwrite(save_root+image_name, img_arr)

if __name__ == '__main__':
    pred_path = './output/diffusionDet_8bit_ela_6/test_lnq/inference/instances_predictions.pth'
    # pred_path = './post_result/0711R50_4Step_uniform_buffer_500box/test_post.pth'
    # pred_path = './test_post.pth'
    data_root = '/home/xinyul/python_exercises/data/lesions/COVID_coco/images/'
    gt_root = "/home/xinyul/python_exercises/data/lesions/COVID_coco/annotation.json"
    save_root = './output/diffusionDet_8bit_ela_6/test_lnq/inference/img_test/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    result = torch.load(pred_path)
    gt = COCO(gt_root)
    # gt_vis(gt, './datasets/8bit/testset/show/')
    imgs=gt.imgs
    filename=[]
    # filename_series=[]
    for i in range(0,len(imgs)):
        filename.append(imgs[i]['file_name'][:-4])
    result = prepocess(result, thr=0.01)
    final_eva=[]
    # final_eva_series=[]
    final_eva.append(filename)
    # final_eva_series.append(filename_series)
    TP, FN, FP, final_eva = eval_frame(result, load_annotations(gt_root), final_eva)
    FPPI = FP/len(imgs)
    # TP_series, FN_series, FP_series, final_eva_series = eval_series(result, load_annotations(gt_root),final_eva_series)
    # AP = TP / (TP + FN)#Sens
    # AR = TP / (TP + FP)#prec 0.7962962962962963
    prec = TP / (TP + FP)
    sens = TP / (TP + FN)
    F1=2*prec*sens/(sens+prec)# 0.7226
    # AP_series = TP_series / (TP_series + FN_series)#sens 0.6615384615384615
    # AR_series = TP_series / (TP_series + FP_series)#prec 0.7962962962962963
    # F1_series=2*AP_series*AR_series/(AP_series+AR_series)# 0.7226
    print('                  TP:{} FP:{} FN:{} \n                  prec:{} sens:{} F1:{} @FPPI:{}\n'
                    .format(TP, FP, FN, prec, sens, F1, FPPI))
    # print('                  TP-series:{} FP-series:{} FN-series:{} \n                  AP-series:{} AR-series:{} F1-series:{}\n'
    #                     .format(TP_series, FP_series,FN_series,AP_series, AR_series,F1_series))
    # data_write('./test_result/frame_eval_r50_wobuffer.xls', final_eva)
    # data_write('./test_result/frame_eval_series_r50_wobuffer.xls', final_eva_series)
    # show_img(result, gt, save_root) 
    show_img_contrast(result, gt, save_root)