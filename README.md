# HATDAR

Hunan activity transition detection and activity recognition in a stream of mmWave sensor data of various human activities by analyzing the inner correlation of mmWave radar data fragments.

- Pocess:  
        - Detect the transition from one activity to another.  
        - Recognizes the new activity type in the data once it detects an activity transition.  
  

**Python version**: This code is in Python3.7

**Package Requirements**: TensorFlow2 numpy sklearn matplotlib

The dataset is borrowed from RADHAR(https://github.com/nesl/RadHAR)

## Data Preoricessubg

#### Step 1 Voxel Normalization
```
python voxel_generator.py --frame 120 --sliding_window 10 --data_path data/raw --data_save data/voxel/
```
* `frame` is the length of a radar segment, default as 120, `sliding_window` default is 10. `data_path` is the raw dataset path, `data_save` is the voxel data storage path.

#### Step 2 Feature Extraction
*In the dataset, each fragment is represented by 120*32*32*16 elements in the spatial domain. An adversarial effect of this data presentation is, however, the size of the whole new dataset, which results in more than 2 TB. To reduce the computation stress we have applied a feature extraction process to the data with a Covolutional Neural Network, which compresses and converts the sptial data into a feature domain.
```
python CNN.py --frame 120 --voxel_path data/voxel/ --model_path model_data/cnn/model.h5 --learning_rate 0.001 --beta_1 0.9 --beta_2  0.999 --batch_size 15 --epochs 50
```
* CNN network for feature extraction
```
python feature_extraction.py --frame 120 --voxel_path data/voxel/ --feature_path data/feature/ --model_path model_data/cnn/model.h5
```
* 'voxel_path' is the directory containing voxel, 'model_path' is the directory of CNN model which used to extract features, and feature data is saved in `feature_path`

#### Step 3 Transition Activity dataset Development
```
python TranAct.py --scope 110 --feature_path data/feature/ --transition_path data/transition/ --dataset train
```



