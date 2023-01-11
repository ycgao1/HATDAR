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

#### Step 2 CNN Training
```
python CNN.py --frame 120 --voxel_path data/voxel --model_path model_data/cnn/model.h5 --learning_rate 0.001 --beta_1 0.9 --beta_2  0.999 --batch_size 15 --epochs 50
```


#### Step 1 Voxel Normalization
```
python voxel_generator.py --frame 120 --sliding_window 10 --data_path data/raw --data_save data/voxel/
```


#### Step 1 Voxel Normalization
```
python voxel_generator.py --frame 120 --sliding_window 10 --data_path data/raw --data_save data/voxel/
```
