# MACHINE LEARNING :radio_button:
# Autoencoder-Neural-Network
Assignment for lesson: [`Software Development for Algorithmic Problems`](https://www.di.uoa.gr/en/studies/undergraduate/261) (DI University of Athens)

## Authors:
[Vasiliki Koumarela](https://github.com/VasiaKoum)
[Charalampos Katimertzis](https://github.com/chariskms)

## Parts
### Part 1 : Convolutional Autoencoder
### Part 2 : Neural Network Classifier (CNN + FC)

## Introduction
This project is about training and evaluation of neural network autocoding of
numerical images digits. Encoder is used to create a neural network of image
classification. First, we construct the autoencoder N1 with these
hyperparameters: Number_of_Layers, Number_of_Filter_Size,
Number_of_Filters_by_Layer, Number_of_Epochs & Number_of_Batch_Size,
we split the dataset in training set & validation set and after the execution,
the model is saved. Then, for the second part, we construct a NN for image
classification (N2). The N1 model is used to form a fully connected NN and an
output layer. The training of N2 is done in two stages to achieve the
reduction of convergence time.

## Requirements
● Python 3 <br>
● Keras <br>
● Numpy <br>
● Matplotlib <br>
● Tensorflow <br>
● Pandas <br>
● Sklearn <br>

## Files
❖ autoencoder.py <br>
❖ classification.py <br>
❖ functions.py (functions for both programs) <br>

**_Metrics and times were measured at Google Colab with specs: CPU x2 @
2.2GHz, RAM:13 GB, GPU: Nvidia Tesla K_**

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part 1
The model was trained with a subset of 60k of black and white images of
digits. The encoder for one layer uses **Conv2D** , **BatchNormalization** and
**MaxPooling2D** (for the first 2 layers). Respectively, the decoder uses Conv2D,
BatchNormalization and **UpSampling** (for the last 2 layers). The program is
using the mirrored architecture, that is symmetric layers in the encoder &
decoder, which is more efficient. **Mean_Squared_error** (for loss →
mean_of_square(y_true - y_pred)) & **RMSprop** (for optimizer → maintaining a
moving average of the square of gradients, dividing the gradient by the root
of this average) were used to compile the model. Also, for each hidden layer,
**REctified Linear Unit** activation function is used except from the last layer in
the decoder that **Sigmoid** activation function is called. No Dropout filter is
used because from the experiments we concluded that the val_loss is much
lower than loss, which is not a normal training and most importantly no
overfitting was observed.

### Execution
``
python autoencoder.py -d dataset
``

The user choices are: 1) Execution with different hyperparameters, 2) Plot of the
error-graphs, 3) Save of the existing model, 4) Exit. If the user chooses (1), then
the user choices are: 1) Layers, 2) Filter_Size, 3) Filters/Layer, 4) Epochs, 5)
Batch_Size. For the graph plots, the program shows a graph with the losses &
the changed hyperparameter. The user, executing the program once, can
change multiple hyperparameters and the graph plots will include all the
changes of the hyperparameters, keeping the initial hyperparameters
constant and each time the selected hyperparameter will be changed.

### Experiments
We executed the autoencoder.py with different values for hyperparameters
and these are the results: <br>
    **● L4_FS3_FL32_Ex_B64:** <br>
    ![L4_FS3_FL32_Ex_B64](https://user-images.githubusercontent.com/44468438/108232979-ff9e9180-714b-11eb-91b3-18ad38c68a16.png) <br>
    **● Lx_FS3_FL16_E100_B64:** <br>
    ![Lx_FS3_FL16_E100_B64](https://user-images.githubusercontent.com/44468438/108233326-55733980-714c-11eb-8a13-5fae785e9a0d.png) <br>
    **● L4_FS3_FL32_E100_Bx:** <br>
    ![L4_FS3_FL32_E100_Bx](https://user-images.githubusercontent.com/44468438/108232877-e3025980-714b-11eb-876a-a8ddb3c792db.png) <br>
    **● L4_FSx_FL32_E100_B64:** <br>
    ![L4_FSx_FL32_E100_B64](https://user-images.githubusercontent.com/44468438/108233087-1a710600-714c-11eb-9b27-c7eea6de0e45.png) <br>
    **● L4_FS3_FLx_E100_B64:** <br>
    ![L4_FS3_FLx_E100_B64](https://user-images.githubusercontent.com/44468438/108233049-0fb67100-714c-11eb-9696-1795c4af0957.png)
    
### Metrics - loss_values.csv
| Layers | Filter_Size | Filters/Layer | Epochs | Batch_Size | Train_Time         | Loss                   | Val_Loss               |
|--------|-------------|---------------|--------|------------|--------------------|------------------------|------------------------|
| 4.0    | 3.0         | 32.0          | 100.0  | 40.0       | 1045.951033115387  | 0.0003012537199538201  | 0.00044053187593817727 |
| 4.0    | 3.0         | 32.0          | 300.0  | 64.0       | 2166.6627881526947 | 0.00021637111785821617 | 0.0004895623424090445  |
| 4.0    | 3.0         | 32.0          | 100.0  | 64.0       | 847.369401216507   | 0.0003372598148416728  | 0.0005164727917872369  |
| 4.0    | 3.0         | 32.0          | 200.0  | 64.0       | 1444.885232925415  | 0.0002477078523952514  | 0.0005514724762178957  |
| 4.0    | 3.0         | 32.0          | 100.0  | 80.0       | 813.5535976886749  | 0.0003667267446871847  | 0.0005788473063148558  |
| 4.0    | 2.0         | 32.0          | 100.0  | 64.0       | 777.8799357414246  | 0.0004566302231978625  | 0.000579037528950721   |
| 4.0    | 3.0         | 64.0          | 100.0  | 64.0       | 1357.034786939621  | 0.0001967029966181144  | 0.0006470134831033647  |
| 4.0    | 3.0         | 16.0          | 100.0  | 40.0       | 698.5567531585692  | 0.0005995309329591691  | 0.0006567330565303563  |
| 5.0    | 3.0         | 32.0          | 100.0  | 40.0       | 2584.629983901977  | 0.00024339786614291367 | 0.0006997981108725071  |
| 4.0    | 3.0         | 16.0          | 100.0  | 64.0       | 484.79444241523737 | 0.00065751833608374    | 0.0007080046343617141  |
| 5.0    | 3.0         | 16.0          | 100.0  | 64.0       | 794.8870048522949  | 0.0005632395623251796  | 0.0009144683135673405  |
| 6.0    | 3.0         | 16.0          | 100.0  | 64.0       | 1863.0545954704285 | 0.0004291314398869872  | 0.0011908371234312654  |
| 4.0    | 4.0         | 32.0          | 100.0  | 64.0       | 1737.8103952407835 | 0.0004900431376881896  | 0.0014112758217379444  |
| 6.0    | 3.0         | 32.0          | 100.0  | 40.0       | 4743.804999351502  | 0.00032031734008342033 | 0.0014694344718009233  |


### Results
From the experiments we observe that the best values ​​for the
hyperparameters are: ​Layers=4, Filter_Size=3, Filters/Layer=32, Epochs ∈
[100,300], Batch_Size ∈ [40,64].​ To choose the values for the hyperparameters
we observe that when layers are increased, the loss plot falls. On the other
hand, validation_loss for layer=4 has the minimum loss (<4 & >4 val_loss is
increasing). Also, for Filter_Size, the best choice is 3 because this value has the
minimum loss & val_loss. Besides that, 3 is usually the one chosen because we
have symmetry with the previous layer pixel and the output, so we will not
have to account for distortions across the layers which happen when using
an even sized kernel. The Filters/Layer catch few of some simple features of
images (edges, color tone, etc) and the next layers are trying to obtain more
complex features based on simple ones. From the error-graphs, the best
value for Filters/Layer is 32 because in this value, the val_loss is the minimum.
For Epochs and Batch_Size, as we increase the epochs, the losses are
decreasing and respectively as we decrease the Batch_Size, the values for
losses are decreasing, as well. From loss_values.csv, we observe that the
Layers and Filters/Layer affect negatively training time and the deeper the
CNN, the higher are the loss values. Finally, Dropout layer isn’t required
because the model isn’t overfitting eg: <br>
**With Dropout: No Dropout:**

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part 2
The classification model splits in two parts. In the first part, we use the
encoder with the trained weights from the autoencoder which we have
already trained. The encoding part encodes the test data in order to make the
1D array that is needed for classification. The second part consists of a fully
connected network layer and an output layer so as to extract the correct
predictions for our labels. The fully connected network for the experiments
uses the softmax activation function. No Dropout filter is used because from
the experiments we concluded that the val_loss is much lower than loss,
which is not a normal training and most importantly no overfitting was
observed.

### Execution
``
python classification.py -d –d <training set> –dl <traininglabels> -t <testset> -tl <test labels> -model <autoencoder h5>
``

The user choices are: 1) Execution with different hyperparameters, 2) Plot of the
error-graphs, 3) Predict the test data. (outputs 2 images of the predicted
labels) 4) Exit. If the user chooses (1), then the user choices are: 1) Layers, 2) Fc
units, 3) Epochs, 4) Batch_Size. For the graph plots, the program shows a
graph with the losses, the accuracies & the changed hyperparameter. The
user, executing the program once, can change multiple hyperparameters
and the graph plots will include all the changes of the hyperparameters,
keeping the initial hyperparameters constant and each time the selected
hyperparameter will be changed.

### Experiments
We executed the classification.py with optimized autoencoder model
**L4_FS3_FL32_E100_B40** and different values for hyperparameters and these are the results: <br>
  **● L4_FC32_Ex_B64: For the training with Epochs 100:** <br>
  **● L4_FCx_E20_B64: For the training with Fc units 256:** <br>
  **● L4_FC32_E20_Bx: For the training with Batch size 512:** <br>
  **● Lx_FC32_E20_Bx: For the training with 6 layers:** <br>

### Metrics - classification_loss_values.csv
| Layers | Fc_units | Epochs | Batch_Size | Train_Time         | Loss                   | Val_Loss              | Accuracy           | Val_Accuracy       |
|--------|----------|--------|------------|--------------------|------------------------|-----------------------|--------------------|--------------------|
| 4.0    | 32.0     | 200.0  | 64.0       | 1518,936660528183  | 0.0002462830452714116  | 0.0021068472415208817 | 0.998770833015442  | 0.9892500042915344 |
| 4.0    | 32.0     | 100.0  | 64.0       | 726.2077898979187  | 0.0004056310572195798  | 0.002265516668558121  | 0.9979583621025084 | 0.9882500171661376 |
| 4.0    | 32.0     | 20.0   | 64.0       | 341.91988229751587 | 0.0011126850731670856  | 0.002528909826651216  | 0.9941874742507936 | 0.9869999885559082 |
| 5.0    | 32.0     | 20.0   | 64.0       | 460.14333486557007 | 0.001573132467456162   | 0.002635737415403128  | 0.9919166564941406 | 0.9862499833106996 |
| 4.0    | 32.0     | 20.0   | 128.0      | 240.39850044250488 | 0.0009345614234916867  | 0.0026041739620268345 | 0.9951458573341372 | 0.9860833287239076 |
| 4.0    | 32.0     | 30.0   | 64.0       | 211.14050102233887 | 0.0008377633057534696  | 0.002835295395925641  | 0.9956666827201844 | 0.9853333234786988 |
| 4.0    | 16.0     | 20.0   | 64.0       | 343.56931710243225 | 0.0010539183858782053  | 0.002811627462506295  | 0.9944375157356262 | 0.9851666688919068 |
| 4.0    | 32.0     | 20.0   | 32.0       | 633.3892841339111  | 0.0018331960309296849  | 0.0030100878793746233 | 0.990666687488556  | 0.984666645526886  |
| 6.0    | 32.0     | 20.0   | 64.0       | 1169.6057088375094 | 0.0015772972255945206  | 0.002966598141938448  | 0.9918333292007446 | 0.9844999909400941 |
| 4.0    | 64.0     | 20.0   | 64.0       | 350.093138217926   | 0.0013049677945673466  | 0.002976583782583475  | 0.9932708144187928 | 0.9844999909400941 |
| 4.0    | 32.0     | 20.0   | 256.0      | 185.44765949249268 | 0.0006138747557997705  | 0.0030035413801670074 | 0.9966250061988832 | 0.984000027179718  |
| 4.0    | 32.0     | 20.0   | 512.0      | 163.74113273620603 | 0.00043937761802226316 | 0.002933959243819117  | 0.9974791407585144 | 0.9837499856948853 |
| 4.0    | 128.0    | 20.0   | 64.0       | 373.89958024024963 | 0.0023482625838369127  | 0.0035958236549049616 | 0.9880416393280028 | 0.9816666841506958 |
| 4.0    | 32.0     | 10.0   | 64.0       | 74.47098183631897  | 0.0018174701835960148  | 0.0038802921772003174 | 0.9904999732971193 | 0.9797499775886536 |
| 4.0    | 256.0    | 20.0   | 64.0       | 419.769193649292   | 0.08766883611679077    | 0.08561211824417114   | 0.5613541603088379 | 0.5716666579246521 |


### Predictions
**Layers 4.0 Fc_units 32.0 Epochs 100.0 Batch_Size 64**
* CLASSIFICATION_REPORT <br>
<img src="https://user-images.githubusercontent.com/44468438/108229341-6326c000-7148-11eb-9195-1a78df646c80.png" width="47%" height="47%"> <br>
* CORRECT_LABELS <br>
![imagecorrect](https://user-images.githubusercontent.com/44468438/108229384-7174dc00-7148-11eb-8ccb-707cf595a0d4.png)
* INCORRECT_LABELS <br>
![imageincorrect](https://user-images.githubusercontent.com/44468438/108229392-733e9f80-7148-11eb-9f36-21d1fa9f400a.png)


### Results
From the experiments, it is observed that the increase of the layers
doesn’t affect the results in a positive way, but it only increases the
training time. Therefore, it would be recommended to keep the value 4.
The increase of the number of nodes of fully connected layer more than
128 nodes results in the increase of loss and the decrease of accuracies.
Consequently, it is beneficial to keep nodes’ value low. So with the
combination of the other hyperparameters we choose 32 nodes for our
predictions. On the other hand, when the epochs increase, accuracies
are improved significantly at the expense of time. That’s why we believe
that it is useful to stay at 100. In the end, the increase of batch size gains
profit on time and an unimportant improvement is observed
concerning the results. We prefer to keep batch size in [64, 128] based
on val_accuracy (more efficient value 64).


