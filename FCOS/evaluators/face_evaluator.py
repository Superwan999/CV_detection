from dataset.faceData import *
import os
import time
import numpy as np
import pickle
import cv2


class FaceAPIEvaluator:
    def __init__(self,
                 input_path,
                 output_path,
                 device,
                 transform,
                 set_type='evaluate',
                 display=False):
        self.input_path = input_path
        self.device = device
        self.transform = transform
        self.set_type = set_type
        self.labelmap = CLASSES
        self.display = display
        self.imgsetpath = input_path
        self.output_dir = self.get_output_dir(output_path, self.set_type)

        self.dataset = FaceDataSet(input_path=self.input_path,
                                   transform=transform)

    @staticmethod
    def get_output_dir(name, phase):
        filedir = os.path.join(name, phase)
        os.makedirs(filedir, exist_ok=True)
        return filedir

    def evaluate(self, net):
        net.val()
        num_images = len(self.dataset)
        self.all_boxes = [[[] for _ in range(num_images)]
                          for _ in range(len(self.labelmap))]

        det_file = os.path.join(self.output_dir, 'detections.pkl')

        for i in range(num_images):
            im = self.dataset.pull_image(i)
            h, w, _ = im.shape

            orig_size = np.array([[w, h, w, h]])
            x = self.transform(im)[0]
            x = x.unsqueeze(0).to(self.device)
            t0 = time.time()

            bboxes, scores, cls_ids = net(x)

            detect_time = time.time() - t0

            bboxes *= orig_size

            for j in range(len(self.labelmap)):
                inds = np.where(cls_ids == j)[0]
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes,
                                   c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                self.all_boxes[j][i] = c_dets

            if i % 500 == 0:
                print(f'image detect: {i + 1}/{num_images}, {detect_time}')

        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print("Evaluating detections")
        self.evaluate_detection(self.all_boxes)
        print("Mean AP:", self.map)

    def get_face_result_file_template(self, cls):
        filename = 'det_' + self.set_type + f'_{cls}.txt'
        filedir = os.path.join(self.output_dir, 'result')
        os.makedirs(filedir, exist_ok=True)
        path_ = os.path.join(filedir, filename)
        return path_

    def write_face_result_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print("Writing {:s} FACE results file".format(cls))
            filename = self.get_face_result_file_template(cls)

            with open(filename, 'wt') as f:
                for ids, input_file in enumerate(self.dataset.file_list):
                    dets = all_boxes[cls_ind][ids]
                    if dets == []:
                        continue

                    for k in range(dets.shape[0]):
                        f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".
                                format(input_file[0], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def parse_rec(self, imagename):
        objects = []
        labelname = imagename.replace("images", "labels").replace("jpg", "txt")
        with open(labelname, 'r') as f:
            labels = f.readlines()
            f.close()
        for label in labels:
            obj_struct = {}
            label = label.strip('\n')
            obj_struct['name'] = 'face'

            image = cv2.imread(imagename)
            h, w = image.shape[:2]
            txt_ = list(map(float, label.split(' ')))
            bboxes = txt_[1:]
            bboxes[0] = bboxes[0] - (bboxes[2] / 2)
            bboxes[1] = bboxes[1] - (bboxes[3] / 2)
            bboxes[2] += bboxes[0]
            bboxes[3] += bboxes[1]

            bboxes[:, 0] *= w
            bboxes[:, 1] *= h
            bboxes[:, 2] *= w
            bboxes[:, 3] *= h
            obj_struct['bbox'] = list(map(int, bboxes))
            objects.append(obj_struct)
        return objects

    def voc_ap(self, rec, prec, use_07_metric=True):
        """
        Compute VOC AP given precision and recall.
        if use_07_metric is true, uses the VOC 07 11 point method (default: True)
        """
        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p // 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelop
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis recall changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def face_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        os.makedirs(cachedir, exist_ok=True)
        cachfile = os.path.join(cachedir, 'annots.pkl')

        # read list of images
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip('\n') for x in lines]

        # for i, imagename in enumerate(imagenames):
        #     R = [obj for obj in recs[imagename]]

        if not os.path.isfile(cachfile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(imagename)

                if i % 100 == 0 and self.display:
                    print(f"Reading label data for {i + 1:d} / {len(imagenames):d}")

            # save
            if self.display:
                print(f"Saving cached label data to  {cachfile:s}")
            with open(cachfile, 'wb') as f:
                pickle.dump(recs, f)
                f.close()
        else:
            # load
            with open(cachfile, 'rb') as f:
                recs = pickle.load(f)
                f.close()

        # extract ground truth for this class
        class_recs = {}
        npos = 0

        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array(x['bbox'] for x in R)
            difficult = np.array([0 for i in range(len(R))]).astype(np.bool)
            det = [False] * len(R)
            npos += sum(~difficult)
            class_recs[imagename] = {
                'bbox': bbox,
                'difficult': difficult,
                'det': det
            }

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
            f.close()
        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(y) for y in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :1]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
                          (BBGT[:, 2] - BBGT[:, 0]) * (
                                  BBGT[:, 3] - BBGT[:, 1] - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)

            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.output_dir, 'annotation_cache')
        aps = []
        for i, cls in enumerate(self.labelmap):
            filename = self.get_face_result_file_template(cls)
            rec, prec, ap = self.face_eval(detpath=filename,
                                           classname=cls,
                                           cachedir=cachedir,
                                           ovthresh=0.5)

            aps += [ap]
            print(f"AP for {cls} = {ap:.4f}")
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if self.display:

            self.map = np.mean(aps)
            print(f'Mean AP = {self.map:.4f}')
            print('~~~~~~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def evaluate_detection(self, box_list):
        self.write_face_result_file(box_list)
        self.do_python_eval()


if __name__ == "__main__":
    pass
