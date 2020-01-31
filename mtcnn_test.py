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

def video_test():
    video_name = './data/test3.mp4'
    pnet = './model_store/pnet_epoch_10.pt'
    rnet = './model_store/rnet_epoch_30.pt'
    onet = './model_store/onet_epoch_30.pt'
    pnet, rnet, onet = create_mtcnn_net(p_model_path=pnet, r_model_path=rnet, o_model_path=onet, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    
    cap = cv2.VideoCapture(video_name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 1
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        # 2
        #bboxs, boxes_align = mtcnn_detector.detect_pnet(frame)
        #bboxs, boxes_align = mtcnn_detector.detect_rnet(frame, boxes_align)
        
        if hasattr(bboxs, 'shape'):
            #print(bboxs.shape, landmarks.shape)
            face_num = bboxs.shape[0]
            for i in range(face_num):
                bbox = bboxs[i, :4]
                bbox = [int(b) for b in bbox]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                
                if 'landmarks' in vars():
                    landmark = landmarks[i, :]
                    landmark = [int(l) for l in landmark]
                    for j in range(int(len(landmark) / 2)):
                        cv2.circle(frame, (landmark[j], landmark[j+1]), int(min(frame.shape[:1]) / 100), (0,0,255), -1)
        
        cv2.imshow('CatFace Detection', frame)    
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break  
    
if __name__ == '__main__':
    test()
    