# IMU and Smartphone Camera Fusion for Knee Adduction and Knee Flexion Moment Estimation During Walking

By Tian Tan and Dianxin Wang

## Exclusive Summary

We xx The paper is in Submission

## Environment

Our environments were:
Python 3.8; Pytorch 1.7.0; Cuda 11.0; Cudnn 8.0.4;

Versions different from ours may still work.

## Example data

The example data file ([example_data.h5](./trained_models_and_example_data/example_data.h5))
contains 10 walking step of 2 subjects. Each subject's data was stored as a 3 dimensional matrix.
The first dimension is walking steps. Its length is 10 because 10 walking steps were provided. The second dimension is
time step from heel-strike - 20 samples to toe-off + 20 samples. Both heel-strike and toe-off are detected using right
foot IMU data. Its length is 152, which is the lenghth of the longest step. Zeros were appended in the end of shorter
steps. The third dimension contains 261 data fields, which are introduced in the following section.

### Data fields

#### Basic information

_force_phase_: ground-truth stance phase;
_body weight_: weight of the subject, being constant for the whole step;
_body height_: height of the subject, being constant for the whole step;

#### External moments of right knee

_EXT_KM_X_: external knee moment in sagittal plane, also known as knee flexion moment;
_EXT_KM_Y_: external knee moment in frontal plane, also known as knee adduction moment;
_EXT_KM_Z_: external knee moment in transverse plane;

#### IMU data
Eight <a href="http://sagemotion.com/" target="_blank"> SageMotion IMUs </a>
(SageMotion, Kalispell, MT, USA) were placed on each subject.
For each IMU, its z-axis is aligned with the body segment surface normal,
y-axis points upwards,
and x-axis being perpendicular to the y and z axes following the right-hand rule.

IMU fields are named in the form of "Measurement(Axis)_Segment".
Measurements are _Accel_: acceleration;
_Gyro_: gyroscope; _Mag_: magnetometer; _Quat_: quaternion.
Segments are _L_FOOT_: left foot; _R_FOOT_: right foot; _R_SHANK_: right shank;
_R_THIGH_: right thigh; _WAIST_: pelvis; _CHEST_: upper trunk;
_L_SHANK_: left shank; _L_THIGH_: left thigh

#### Joints detected from camera data
Two <a href="https://www.apple.com/shop/buy-iphone/iphone-11" target="_blank"> iPhone 11 </a> (Apple Inc., Cupertino, CA, USA)

were placed on the right side (90 degree from walking direction) 
and back of the subject (180 degree from walking direction). 
Videos recorded by these two cameras were processed by
<a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose" target="_blank">OpenPose</a> [1], a body keypoint detection library,
a body keypoint detection library. 2D positions of left/right shoulders, left/right hip, mid-hip, left/right knees,
left/right ankles, and left/right heels detected by OpenPose's
<a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/18de3a0010dd65484b3eb357b5c3679c9a2fdf43/doc/02_output.md" target="_blank">Body_25</a> model.

2D joint position fields are named in the form of "Joint_Axis_Camera".
For example, _LShoulder_x_90_ means pixel position (1080 pixel in total) of left shoulder joint in horizontal direction of right camera view.
_RShoulder_y_180_ means pixel position (1920 pixel in total) of right shoulder joint in vertical direction of back camera view.

#### Force plate measurements
Ground reaction force (GRF) and center of pressure (CoP) were collected with an instrumented treadmill 
with two split belts (Bertec Corp., Worthington, OH, USA).
CoP are calibrated with marker trajectories that the left-back corner was the origin of both force plates and optical motion capture.
The z-axis is aligned with the vertical direction pointing upwards,
y-axis is aligned with the walking direction
(in general aligned with anterior-posterior direction) pointing forwards,
and x-axis being perpendicular to the y and z axes following the right-hand rule
(in general aligned with medio-lateral direction).

#### Marker trajectories from optical motion capture
32 markers were placed on each subject and collected by an optical motion capture system (Vicon, Oxford Metrics Group, Oxford, UK).
Markers' field name are in the form of "MarkerName_Axis".
Marker names' corresponding body landmarks are visualized in the following figures.
![img.png](figures/readme_fig/markers_anterior_view.png xxx)
![img.png](figures/readme_fig/markers_posterior_view.png xxx)


## Use KAM and KFM estimation models
### Trained models
Six models are provided:
Two IMU-based models (e.g. [IMU_based_KAM.pth](./trained_models_and_example_data/IMU_based_KAM.pth)),
two camera-based models (e.g. [camera_based_KAM.pth](./trained_models_and_example_data/camera_based_KAM.pth)), and
two IMU & camera fusion model (e.g. [fusion_KAM.pth](./trained_models_and_example_data/fusion_KAM.pth)).

### Running the code


## References
[1] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh,“Openpose: Realtime multi-person 2d pose estimation
using part affinityfields,”IEEE Transactions on Pattern Analysis and Machine Intelligence,vol. 43, no. 1, pp. 172–186,
2019

