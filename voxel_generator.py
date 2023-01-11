# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:08:45 2020

@author: Yichen Gao
"""



import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

sub_dirs=['boxing','jack','jump','squats','walk']

def coordinate2array(x, y, z, intensity, x_point, y_point, z_point):
    #boundary
    x_max=np.max(x)
    x_min=np.min(x)   
    y_max=np.max(y)
    y_min=np.min(y)    
    z_max=np.max(z)
    z_min=np.min(z)
    
    #x_step=(x_max-x_min)/x_point
    #y_step=(y_max-y_min)/y_point
    #z_step=(z_max-z_min)/z_point
    
    #change
    X=x_point*(x-x_min)/(x_max-x_min)
    Y=y_point*(y-y_min)/(y_max-y_min)
    Z=z_point*(z-z_min)/(z_max-z_min)
    
    #pixel=np.zeros([2*x_point, y_point, z_point])
    pixel=np.zeros([x_point, y_point, z_point])
    
    
    for i in range(len(X)):
        if X[i]==0:
            x_c=0
            y_c=0
            z_c=0
        else:
            x_c=int(round(X[i])-1)
            y_c=int(round(Y[i])-1)
            z_c=int(round(Z[i])-1)
        pixel[x_c,y_c,z_c]=pixel[x_c,y_c,z_c]+1
        #pixel[x_c,y_c,z_c]=pixel[x_c,y_c,z_c]+intensity[i]
        #pixel[x_c+x_point,y_c,z_c]=pixel[x_c+x_point,y_c,z_c]+intensity[i]
        
    del x, y, z, intensity
        
    return pixel

def get_data(file_path, total_frame, sliding):
    
    with open(file_path) as f:
        lines=f.readlines()
        
    wordlist=[]
    for line in lines:
        for word in line.split():
            wordlist.append(word)
      
    #extract x,y,z
    frame_num_count=-1
    frame_num=[]
    x=[]
    y=[]
    z=[]
    intensity=[]
    for i in range(len(wordlist)):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count+=1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    #list2array
    frame_num=np.asarray(frame_num)
    x=np.asarray(x)
    y=np.asarray(y)
    z=np.asarray(z)
    intensity=np.asarray(intensity)
    
    #datatype
    frame_num = frame_num.astype(np.int)
    x = x.astype(np.float)
    y = y.astype(np.float)
    z = z.astype(np.float)
    intensity=intensity.astype(np.float)
    
    #print(len(frame_num))
    #frame
    data=dict()
    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i],y[i],z[i],intensity[i]])
        else:
            data[frame_num[i]]=[]
            data[frame_num[i]].append([x[i],y[i],z[i],intensity[i]])
    
    #dict2pixels
    pixels=[]
    for i in data:
        data2=data[i]
        data2=np.asarray(data2)
        
        x_c=data2[:,0]
        y_c=data2[:,1]
        z_c=data2[:,2]
        intensity_c=data2[:,3]       
        
        pix = coordinate2array(x_c, y_c, z_c, intensity_c, 16, 32, 32)
        pixels.append(pix)
    pixels=np.asarray(pixels)
    
    #sild window
    #total_frame=60;
    #silding=15;
    i=0
    data_sildwindow=[]
    while (i+total_frame)<= pixels.shape[0]:
        data_sildwindow.append(pixels[i:i+total_frame,:,:,:])
        i=i+sliding
    
    if pixels.shape[0]%sliding!=0:
        data_sildwindow.append(pixels[pixels.shape[0]-total_frame:pixels.shape[0],:,:,:])
    
    data_sildwindow=np.asarray(data_sildwindow)
    #return data, pixels
    
    del frame_num, x, y, z, intensity, wordlist, pixels, data, 
    
    return data_sildwindow


def parse_RF_files(parent_dir, sub_dirs, total_frame, sliding, file_ext='*.txt'):
    print(sub_dirs)
    #features =np.empty((0, total_frame, 20, 32, 32) )
    features =np.empty((0, total_frame, 16, 32, 32) )
    labels = []

    for sub_dir in sub_dirs:
        files=sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
        for fn in files:
            print(fn)
            print(sub_dir)
            train_data = get_data(fn, total_frame, sliding)
            features=np.vstack([features,train_data])


            for i in range(train_data.shape[0]):
                #labels.append(action[sub_dir])
                labels.append(sub_dir)
            print(features.shape,len(labels))
            

            del train_data
            #gc.collect()
    labels=np.asarray(labels)

    return features, labels

#data=get_data('action_radar\\data\\Train\\boxing\\20_boxing_1.txt', total_frame=60)
#data_sildwindow=get_data('F:\RadHAR\data\Train\\boxing\\20_boxing_1.txt')
##a=np.vstack([features,pixels])
#action={'boxing':0, 'jack':1, 'jump':2, 'squats':3, 'walk':4}  
def extract(frame, sliding, parent_dir, voxel_path):
    
    for sub_dir in sub_dirs:
        features, labels = parse_RF_files(parent_dir,[sub_dir], frame, sliding)
        Data_path = voxel_path + sub_dir
        if features.shape[0]==0:
            print("no ",sub_dir," files")      
        else:
            np.savez(Data_path, features,labels)
            print(sub_dir, features.shape)
        del features, labels

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=120, type=int)
    parser.add_argument("--sliding_window", default=10, type=int)
    parser.add_argument("--data_path", default='data/raw')
    parser.add_argument("--data_save", default='data/voxel/')
    args = parser.parse_args()
    
    parent_dir = args.data_path
    voxel_path = args.data_save
    frame = args.frame
    sliding = args.sliding_window

    return extract(frame, sliding, parent_dir, voxel_path)
    
    
if __name__ == "__main__":
    main()