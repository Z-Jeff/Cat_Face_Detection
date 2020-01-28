import cv2
import numpy as np
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face

def test():
    pnet = './model_store/pnet_epoch_10.pt'
    rnet = './model_store/rnet_epoch_10.pt'
    onet = './model_store/onet_epoch_10.pt'
    
    pnet, rnet, onet = create_mtcnn_net(p_model_path=pnet, r_model_path=rnet, o_model_path=onet, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./test.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    '''
    boxes, boxes_align = mtcnn_detector.detect_pnet(img)
    boxes, boxes_align = mtcnn_detector.detect_rnet(img, boxes_align)
    landmark_align =np.array([])
    boxes_align_ =np.array([])
    boxes_align_, landmark = mtcnn_detector.detect_onet(img, boxes_align)
    '''
    bboxs, landmarks = mtcnn_detector.detect_face(img)
    #print(bboxs, landmarks)
    
    # print box_align
    save_name = 'result.jpg'
    
    #vis_face(img_bg,boxes, None, save_name)
    vis_face(img_bg, bboxs, landmarks, save_name)
    
    
if __name__ == '__main__':
    test()
    