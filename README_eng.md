# 画像認識

# 英語 / English

## NEURAL NETWORK

画像認識はConvoluted Neural Network（CNN）というアルゴリズムを使えば結構きちんと結果ができす。　ですので今月の勉強はCNNに関してだった。

## TRANSFER LEARNING

Pre trained models~
TODO

## HAR-CASCADE

Har-cascadeは画像検出ができるアルゴリズムです。Neural Networksに比べてそんなに画像認識ができないんです。
Neural Networksはたくさん色々な人の顔の写真の中で写真は誰かの顔どうじて分けることまでできて素晴らしいが, Har-cascadeは画像の中に顔があるかどうか、どこにあることまで分かることがでくる。 だけど、それも使えることです. データのデータクリーニングで, 顔の場所だけ分かったらだけなら便利だから, それ使う.

一つのHar-FilterはWeak Filterです。Weakの意味は、ぎりぎり無作為に比べていいです。 だからHar-FilterをCascadeする。（重ねる）　　

始めは一番よく偽陽性がでるHar-filterで、合わない部分を捨てる。
二回目は、二番目偽陽性が出るFilterを使って、合わない部分を捨てる。
繰り返して
繰り返す。
最後のところは、もっと強いFilterでする。あのFilterはもちろん遅いが、残り部分は少なくてきたから大丈夫はずです。

![](readme_images/har_example.png)

1と２には、フィルターと写真の部分は似合わない。Nに、するべきように似合うです。Mには、間違って似合うです。Mのことがあるから、Har-Filterはたくさん重ねにゃくちゃいけないと。


****************

## TENSORFLOW

TensorFlowはGoogleに作られたAIのAPIです。TensorFlowでAPIのことはもっと簡単に作られそうです。TensorFlowは結構新しいけど早く人気になっている。TensorFlowはCPUかNVIDIA　CUDAで仕事ができる。
Tensorflowの中でたくさん人気や便利AIやMachine Learningアルゴリズムもう作られたから早く自分のプロジェクトで使える。あと、自分のアルゴリズムもう、Frameworkで書けますから何でもできそうである。

### TENSORFLOW INSTALLATION

Windowsの場合、Python3.5が必要である。Python2.7は使えないことです。
1. Python 3.5
2. Microsoft Visual C++ 2015 Redistributable Update 3 (x64 version)
3. CUDA 8 			https://developer.nvidia.com/cuda-downloads
  * Windows PATHとCUDAの“Bin”DIRをつなぐ
4. cuDNN v5.1  		https://developer.nvidia.com/cudnn
  * cuDNNの.zipの中から
&nbsp;&nbsp;"cuda\bin\cudnn64_5.dll" 　	→　"～\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\"
&nbsp;&nbsp;"cuda\include\cudnn.h" 　	→　"～\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
&nbsp;&nbsp;"cuda\lib\x64\cudnn.lib" 　	→　"～\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\"
5. Tensorflow
```bash
pip3 install --upgrade tensorflow-gpu
```
Hello　World	：　
>import tensorflow as tf
> hello = tf.constant('Hello, TensorFlow!')
>sess = tf.Session()
>print(sess.run(hello))

## OPENCV

OpencvはPython2.7のLibraryです。 Opencvで画像やビデオに関して色々なことができます。Machine Learningのことができそうけど、わたしの意見はできるだけTensorflowを使った方がいい. だからOpenCVはウエブカメラからTraining dataを集めるようにだけ使います。

### OPENCV INSTALLATION

OpenCVはPython2.7のAPIだけど下のURLでPython3で使えるバージョンがあります。
1. python 3: 　http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
```bash
pip install opencv_python-3.2.0-cp35-cp35m-win32.whl
```

## ======　Execution ========

> python

> python

> python

### 詳しく説明

### (1) Gather data manually.
- Gather photos as data and place each them in apporopriate to the class subfolders. For examples photos of a person smiling in the subfolder "happy", whilst photos of a person crying in the subfolder "sad".
- This process can be done manually, or with a webcam and the help of "photoAquisition.py"

Using PhotoAquisition.py

1.  > python PhotoAcquisition_3.py C:\datasets\test Happy Sad Scared --number_of_samples 100 --photo_interval 0.5
2. The command above will save 100 photos, 2/second into the c/dataset/test directory.
3. The classes will be "happy", "sad", "scared" and thus those subfolders will contain the relevant photos.
4. The program will prompt for the appropriate images (in this example, a pose) on screen.
5. You can optionally specifiy a Har-Classifier .xml file, which will automattically crop around the target object.

#### Use a Har-Classifier

1. add the argument --cascade_classifier {path to .xml}
2. the program will now auto-crop the webcam pictures.
 - If no pictures are saving, then the har-cascaade is failing to locate any target objects on-screen.

#### Make a Har-classifier to simpliy training data collection.

 1. To make using the web camera to gather photos easier, can first train a har-cascaade.
 1. browse www.image-net.org/ for negative images (images without the object). Want about 1000. eg. http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09403734
 1. paste the links into the "imgnet/neg/links.txt" file
 1. run TrainHar.py in neg mode to download the negative images
    > python TrainHar.py --mode neg

    (takes a long time ...)
 1. sort the images by size and delete any rubbish/corrupt images.
 1. run script to list file names (imgnet/neg/negatives.txt)
    > python TrainHar.py --mode negfiles

 1. Manually gather between 1 to 15 positive images (with the target), best is white background. place in "imgnet/pos"

        positive images that are less than 100x100 pixels will be discarded.

 1. After collecting positive images, resize, greyscale and set ROI for each image with Clean_Postives.py
 1. Run TrainHar.py in pos mode to generate training images. The images are just the positive images distorted and then superimposed over the negative images randomly.
     > python TrainHar.py --mode pos

     images are in "imgnet/traincln/"

 1. Train the har-classifier.

     ??? todo ???

 1. To use the newly created classifier, refer to (2) Use a Har-Classifier.

### (2) train classification layer

 + Transfer Learningということを使うから、割に時間かからないが、もちろんトレーニングデータがあればあるほど時間かかる。

 1. python retrain.py --image_dir training_images
 2. View statistics of the training in order to run again with better settings (optional)
    ```bash
    tensorboard: tensorboard --logdir /temp/retrain_logs
    ```
    (browser) http://localhost:6006

    ![](readme_images/retrain.png)

### (3) Classify an image

1. pass a jpeg image to teh classifier (inception only works on jpeg)
    ```bash
    python try-retrain.py test-images\image1.jpg
    ```
2.  The results are a probability for each class.

    ![](readme_images/try-retrain.png)

## ====== Arguments ====

### PhotoAcquisition_3.py
トレーニングのため写真を作る

例　> python PhotoAcquisition_3.py D:\Luke\code\plurals_transferLearning\train-images-faces\hss Happy Sad Scared --number_of_samples 100 --photo_interval 1 --cascade_classifier D:\Luke\code\plurals_transferLearning\har-cascaades\haarcascade_frontalface_default.xml

Arguments:
+ 保存Dir
+ クラス
+ --number_of_samples: クラスずつ何枚写真
+ --photo_interval: 写真を取る間に何分を休む
+ --har: Har-cascadeのDir

### retrain.py
Classification LayerをトレーニングしてClassification layerを作る。

例　> python retrain.py --image_dir train-images-faces-hss-cropped

Arguments:
+ --image_dir: トレーニングデータ（サブフォルダはクラスの名前）。
+ --output_graph： グラフを保存するDIR. デフォルトは "../retrain_graph/output_graph.pb"です。
+ --output_labels: グラフのクラスの名前を保存するDIR. デフォルトは'../retrain_graph/output_labels.txt'です。
+ --summaries_dis: TensorLogのためにLOGのファイルのDIR. デフォルトは'../retrain_graph/retrain_logs'です。
+ --how_many_training_steps:トレーニングは何回で終わる　.デフォルトは4000です。
+ --learning_rate:　Learing rateはどのぐらいWeightを変更する、前のIterationのエラーに沿って.　デフォルトは0.1です。
+ --testing_percentage: テストのデータは何％がトレーニングにとして使う.　デフォルトは10です。
+ --validation_percentage: テストのデータは何％がヴァリデーションにとして使う.　デフォルトは10desu。
+ --eval_step_interval: トレーニングとヴァリデーションの間はどのくらいことです .デフォルトは10です。
+ --train_batch_size: .デフォルトは100です。
+ --test_batch_size: .デフォルトは1です。
+ --validation_batch_size:
+ --print_misclassified_test_images:
+ --model_dir: 作られたグラフの他のファイルを保存するDIR。デフォルトは'../retrain_graph/imagenet'です。
+ --bottleneck_dir: 。デフォルトは'../retrain_graph/bottleneck'です。
+ --final_tensor_name: 新しい作ったグラフの新しいClassification layerの名前です。デフォルトは'final_result'です。
+ --flip_left_right: トレーニングデータを半分が無作為にミラーイメージさせるかどうか選択です。デフォルトは 0 です (つまりしないことです)
+ --random_scale: トレーニングデータの画像のサイズが何％無作為に大きくする。デフォルトは０パーです。
+ --random_brightness: 無作為に選んだトレーニングデータが画像の明りさを変更する。デフォルトは０パーです。

### try-retrain.py
写真をClassifyする. 入力する写真はどのクラスに一番似合うのを出力です。

例　> python try-retrain.py test-images-faces\image1.jpg

args:
+ image_path: classifyをしたい画像のDIRです.
+ har:  Har-cascadeのDir (必要はないがトレーニングで使ってたら使った方がいい)

## 出力

TODO

