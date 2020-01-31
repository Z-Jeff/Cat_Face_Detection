# Cat face detection using MTCNN

# results:
![](https://github.com/Z-Jeff/Cat_Face_Detection/blob/master/result.png)


# Test an image
  * run > python mtcnn_test.py
 
# Training data prepraring
  * download [cat face dataset for landmark](https://static.leiphone.com/cat_face.zip), then unzip it into ./data_set/original/
    * run > python ./anno_store/tool/transform.py change train.csv into .txt(anno_train.txt)

# Training
  * preparing data for P-Net
    * run > python mtcnn/data_preprocessing/gen_Pnet_train_data.py
    * run > python mtcnn/data_preprocessing/assemble_pnet_imglist.py
  * train P-Net
    * run > python mtcnn/train_net/train_p_net.py
    
  * preparing data for R-Net
    * run > python mtcnn/data_preprocessing/gen_Rnet_train_data.py (maybe you should change the pnet model path)
    * run > python mtcnn/data_preprocessing/assemble_rnet_imglist.py
  * train R-Net
    * run > python mtcnn/train_net/train_r_net.py
  
  * preparing data for O-Net
    * run > python mtcnn/data_preprocessing/gen_Onet_train_data.py
    * run > python mtcnn/data_preprocessing/gen_landmark_48.py
    * run > python mtcnn/data_preprocessing/assemble_onet_imglist.py
  * train O-Net
    * run > python mtcnn/train_net/train_o_net.py
    
 # Citation
   [DFace](https://github.com/kuaikuaikim/DFace)
   [mtcnn-pytorch](https://github.com/Sierkinhane/mtcnn-pytorch)
