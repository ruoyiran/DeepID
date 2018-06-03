# DeepID implementation
**Reference:** [Deep Learning Face Representation from Predicting 10,000 Classes](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)
### DeepID Network
![DeepID_Network](https://github.com/Ruoyiran/DeepID/blob/master/images/DeepID_Network.png)
### DeepID Node Inner Structure
![DeepID_Inner_Structure](https://github.com/Ruoyiran/DeepID/blob/master/images/DeepID_Inner_Structure.png)
### Results
* Accuracy
![Accuracy](https://github.com/Ruoyiran/DeepID/blob/master/images/Accuracy.png)
* Loss
![Loss](https://github.com/Ruoyiran/DeepID/blob/master/images/Loss.png)
### Experiments
Best val accuracy: 99.39%

Best test accuracy: 97.05%
### How to run
* Download YoutubeFaces
  * Download: [aligned_images_DB.tar.gz](https://www.cs.tau.ac.il/~wolf/ytfaces/index.html#download)
  * Extract 'aligned_images_DB.tar.gz' to folder
* Data preprocessing
  * run 'python crop.py', crop images and save to disk
  * run 'python split.py', split train_set, valid_set and test_set
  * run 'python convert_images_to_tfrecords.py', convert all datasets to tfrecord format
* Training
  * run 'python train.py'
* Testing
  * run 'python test.py'
