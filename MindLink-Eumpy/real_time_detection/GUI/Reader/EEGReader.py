import mne
import numpy as np
from real_time_detection.EEG import EEG_feature_extract
from real_time_detection.GUI.EmotivDeviceReader import EmotivDeviceReader

# emotiv_reader = EmotivDeviceReader()


# class EEGReader
class EEGReader:
    '''
    This class is used to return the EEG data in real time.
    Attribute:
        raw_EEG_obj: the data for file input. MNE object.
        timestamp: the current time. how much second.
        features: the EEG features.
    '''

    def __init__(self, input_type, file_path=None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the 'Emotiv insight' device.
        '''
        self.emotiv_reader = EmotivDeviceReader()
        self.emotiv_reader.start()
        self.input_type = input_type
        if self.input_type == 'file':
            self.raw_EEG_obj = mne.io.read_raw_fif(file_path, preload=True)
            max_time = self.raw_EEG_obj.times.max()
            self.raw_EEG_obj.crop(28, max_time)
            self.raw_EEG_obj
            self.timestamp = 0.

            cal_raw = self.raw_EEG_obj.copy()
            cal_raw = EEG_feature_extract.add_asymmetric(cal_raw)

            self.features = EEG_feature_extract.extract_average_psd_from_a_trial(cal_raw, 1, 0.)
        else:
            # TODO: read EEG from devic
            self.timestamp = 0.
            pass

    def get_EEG_data(self):

        '''
        Return:
            EEG data: the EEG data
            timestamp: the current timestamp
        '''

        '''
        ValueError: cannot reshape array of size 0 into shape (0,5,newaxis)
        '''
        if self.input_type == 'file':
            sub_raw_obj = self.raw_EEG_obj.copy().crop(self.timestamp,
                                                       self.timestamp + 1.)
            self.timestamp += 1.

            show_raw_obj = sub_raw_obj.copy().pick_channels(['AF3', 'AF4', 'T7', 'T8', 'Pz'])

            return show_raw_obj.get_data(), self.timestamp - 1., None
        else:
            self.timestamp += 1.
            data_list = self.emotiv_reader.get_data()
            PSD_feature = np.array(data_list)
            PSD_feature = PSD_feature.reshape(PSD_feature.shape[0], 5, -1)  # error
            '''
            error:
            line 236, in get_EEG_data
    PSD_feature = PSD_feature.reshape(PSD_feature.shape[0], 5, -1)
    ValueError: cannot reshape array of size 0 into shape (0,5,newaxis)
    '''
            #             raw_EEG = np.mean(PSD_feature, axis = 2)
            raw_EEG = PSD_feature[:, :, 4]
            raw_EEG_fill = np.zeros((257, 5))
            for i in range(raw_EEG.shape[0]):
                start = i * (int(257 / raw_EEG.shape[0]))
                end = (i + 1) * (int(257 / raw_EEG.shape[0])) if i != raw_EEG.shape[0] - 1 else raw_EEG_fill.shape[0]
                raw_EEG_fill[start:end, :] = raw_EEG[i]
            return raw_EEG_fill.T, self.timestamp - 1., PSD_feature
# class EEGReader