#CNN_expression_biseline.h5

a CNN model for predicting discrete emotion based on human faces
inputs (?, 48, 48, 1)
output (?, 7), represets 7 different emotion in FER-2013 dataset

#CNN_face_regression.h5

a CNN model for predicting continuous emotion based on human faces
inputs (?, 48, 48, 1)
output (?, 2), represets the value of valence and arousal

#continuous_to_discrete.h5

a full connect neutral network for map continuous emotion to discrete emotion.
inputs (?, 2), valence and arousal
outputs (emotion_state, emotion_strength)
emotion state: (?, 16), 16 different emotion in emotion wheel
emotion_strength: (?, 1)

#EEG_mean.npy

A matrix contains the mean value of every feature in EEG data, it was used 
for preprocessing the EEG data.
shape: (85, )

#EEG_std.npy

A matrix contains the standard error value of every feature in EEG data, it 
was used for preprocessing the EEG data.
shape: (85, )

#LSTM_EEG_regression.h5

a LSTM model for predicting continuous emotion base on human EEGs
input shape: (?, 10, 85), 10 time unit, in each time unit, there are 85 features.
output shape: (?, 2), represents the value of valence and aorusal

#enum_weights.npy

a tuple saved the weight of valence and arousal for EEG and faces.
shape: (weight of valences, weight of arousal)
weight of valences: a scalar
weight of arousals: a scalar
use it as weight * face_output + (1-weight) * EEG_output