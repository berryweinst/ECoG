import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import cv2
from PIL import Image
import h5py as h5
import pickle
import gc



class video_data_exp(object):

    def __init__(self, num_of_labels=4,
                 data_file='/Users/berryweinstein/studies/ECoG/video_data/gamma_data.csv',
                 drifts_file='/Users/berryweinstein/studies/ECoG/video_data/drifts.csv',
                 fixations_file='/Users/berryweinstein/studies/ECoG/video_data/fixations.csv',
                 samples_file='/Users/berryweinstein/studies/ECoG/video_data/samples.csv',
                 video_file='/Users/berryweinstein/studies/ECoG/video_data/BeFe-1-recording.avi',
                 hdf5_file='/Users/berryweinstein/tensorflow/TF_FeatureExtraction/features.h5',
                 cnn_model = 'resnet_v1_50',
                 ecog_interval=50,
                 ecog_offset=500,
                 sync_val=14325,
                 cv_image_shape = [720, 960],
                 image_extract=False,
                 znorm_ecog=False):
        self.data_file = data_file
        self.drifts_file = drifts_file
        self.fixations_file = fixations_file
        self.samples_file = samples_file
        self.video_file = video_file
        self.hdf5_file = hdf5_file
        self.ecog_interval = ecog_interval
        self.ecog_offset = ecog_offset
        self.sync_val = sync_val
        self.num_of_labels = num_of_labels
        # self.postData = np.array([])
        # self.postLabels = np.array([])
        self.image_extract = image_extract
        self.cv_image_shape = cv_image_shape
        self.cnn_model = cnn_model
        self.samples_cnt = 0
        self.znorm_ecog = znorm_ecog


    def extract_image_from_video_at_time(self, vidcap, ms_time):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, ms_time)      # Go to the 1 sec. position
        ret,image = vidcap.read()                   # Retrieves the frame at the specified second
        return image



    def extract_images(self):
        sample_size = self.ecog_offset

        print("Reading ECoG activations from %s" % (self.data_file))
        all_ecog_data = pd.read_csv(self.data_file, sep=',', header=None, float_precision='%.3f', low_memory=False)
        print("Done.")
        all_ecog_data = np.array(all_ecog_data)  ## all that is marked as 'good_e'
        if (self.znorm_ecog):
            for idx, e in enumerate(all_ecog_data):
                all_ecog_data[idx] = np.divide((e - np.mean(e)), np.std(e))
        postData = np.empty([0, all_ecog_data.shape[0], int((self.ecog_offset - self.ecog_interval + 10)/10)])
        elec_tl_avg = np.zeros((all_ecog_data.shape[0], sample_size))

        drifts = np.array(pd.read_csv(self.drifts_file, sep=',', header=None, low_memory=False))
        fixations = pd.read_csv(self.fixations_file, sep=',', header=0, low_memory=False);
        samples = pd.read_csv(self.samples_file, sep=',', header=0, low_memory=False);
        samples['Time'] = (samples['Time'] - samples['Time'][0]) / 1000.0

        print("Reading video file %s" % (self.video_file))
        vidcap = cv2.VideoCapture(self.video_file)
        print("Done.")
        fixations_idx = np.where(fixations['Fixation Duration [ms]'] > 150)[0]
        idx2 = 0
        print("Calculating the target vectors by averagind the ECog")
        for i, idx in enumerate(fixations_idx):
            if (idx2 % 500 == 0 or i == len(fixations_idx) - 1):
                print("Done: [%s] frames" % (idx2))
            fix_start = fixations['Fixation Start [ms]'][idx]
            fix_end = fixations['Fixation End [ms]'][idx]
            fix_mean = (fix_start + fix_end) / 2
            tm = np.argmin(np.abs(samples['Time'] - fix_mean))
            if (self.image_extract):
                fc = samples['Frame'][tm + 2]
                frame_time = float(fc[9:11]) / 31.0 * 1000 + float(fc[6:8]) * 1000 + float(fc[3:5]) * 60 * 1000
                frame_time = int(frame_time)
                image = self.extract_image_from_video_at_time(vidcap, frame_time)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image.shape)

            x_pixel = np.int(samples['B POR X [px]'][tm + 2])
            y_pixel = np.int(samples['B POR Y [px]'][tm + 2])
            if (y_pixel not in range(30, self.cv_image_shape[0] - 30) or x_pixel not in range(30, self.cv_image_shape[1] - 30)):
                continue
            else:

                ecog_time = int((fix_start + drifts[0][idx] * 1000 + self.sync_val) * 512 / 1000)

                seg_data = np.empty([all_ecog_data.shape[0], 0])
                for seg in range(0, self.ecog_offset - self.ecog_interval + 10, 10):
                    all_ecog_seg = all_ecog_data[:, ecog_time + seg : ecog_time + seg + self.ecog_interval]
                    ecog_mean = np.mean(all_ecog_seg, axis=1, dtype=float)
                    seg_data = np.column_stack((seg_data, ecog_mean))

                postData = np.concatenate([postData, seg_data.reshape((1, seg_data.shape[0], seg_data.shape[1]))])
                elec_tl_avg = np.add(elec_tl_avg, all_ecog_data[:, ecog_time : ecog_time + sample_size])
                idx2 += 1
                if (self.image_extract):
                    image_out = np.zeros([256, 256, 3])
                    image_out.fill(np.mean(image))
                    y_start = int(max(0, y_pixel - 256 / 2 + 1))
                    y_end = int(min(y_pixel + 256 / 2, image.shape[0]))
                    x_start = int(max(0, x_pixel - 256 / 2 + 1))
                    x_end = int(min(x_pixel + 256 / 2, image.shape[1]))
                    # image_out[0:y_end-y_start, 0:x_end-x_start,:] = image[y_start:y_end,x_start:x_end,:]

                    im = Image.fromarray(image[y_start:y_end, x_start:x_end, :])
                    file = "/Users/berryweinstein/studies/ECoG/video_data/frames/frame_" + str(idx) + ".jpg"
                    im.save(file)
        # print("Loading HDF file of the Resnet50 feature maps")
        # f = h5.File("features.h5", "r")
        data_dict = {}
        data_dict['src'] = postData
        self.samples_cnt = postData.shape[1]
        # data_dict['target'] = {}
        # data_dict['target']['conv1'] = np.array(f[self.cnn_model]['conv1'], dtype=float)
        # data_dict['target']['block1'] = np.array(f[self.cnn_model]['block1'], dtype=float))
        # data_dict['target']['block2'] = np.array(f[self.cnn_model]['block2'], dtype=float))
        # data_dict['target']['block3'] = np.array(f[self.cnn_model]['block3'], dtype=float))
        # data_dict['target']['block4'] = np.array(f[self.cnn_model]['block4'], dtype=float))
        print("Shape of the source: ", data_dict['src'].shape)
        # print("Shape of the target conv1: ", data_dict['src'].shape)
        # print("Shape of the target block1: ", data_dict['target']['block1'].shape)
        # print("Shape of the target block2: ", data_dict['target']['block2'].shape)
        # print("Shape of the target block3: ", data_dict['target']['block3'].shape)
        # print("Shape of the target block4: ", data_dict['target']['block4'].shape)
        print("Saving the Pickle dictionary")
        pickle.dump(data_dict['src'], open("data_dict_src.p", "wb"))
        print ("data_dict_src.p... saved")
        # pickle.dump(data_dict['target']['conv1'], open("data_dict_target_conv1.p", "wb"))
        # print ("data_dict_target_conv1.p... saved")
        # pickle.dump(data_dict['target']['block1'], open("data_dict_target_block1.p", "wb"))
        # print ("data_dict_target_block1.p... saved")
        # pickle.dump(data_dict['target']['block2'], open("data_dict_target_block2.p", "wb"))
        # print ("data_dict_target_block2.p... saved")
        # pickle.dump(data_dict['target']['block3'], open("data_dict_target_block3.p", "wb"))
        # print ("data_dict_target_block3.p... saved")
        # pickle.dump(data_dict['target']['block4'], open("data_dict_target_block4.p", "wb"))
        # print ("data_dict_target_block4.p... saved")
        elec_tl_avg = np.divide(elec_tl_avg, idx2)
        print("Saving the Pickle time locked averages")
        pickle.dump(elec_tl_avg, open("elec_tl_avg.p", "wb"))
        print("elec_tl_avg.p... saved")
        # return elec_tl_avg
        # return 1



exp = video_data_exp(znorm_ecog=True)
exp.extract_images()
gc.collect()





# model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
#
# activations_dict = {}
# for frame_idx, img_path in enumerate(glob.iglob('/Users/berryweinstein/studies/ECoG/video_data/frames/*.jpeg')):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.resnet50.preprocess_input(x)
#     preds = model.predict(x)
#     activations_dict[frame_idx] = {}
#     activations_dict[frame_idx][0] = model.layers[1].activation(x)
#     activations_dict[frame_idx][0] = model.layers[1].activation(x)
#     activations_dict[frame_idx][0] = model.layers[1].activation(x)
#     print('Predicted frame_idx (%d): %s' % (frame_idx, tf.keras.applications.resnet50.decode_predictions(preds)[0][0][1]))







# print('Predicted:', tf.keras.applications.resnet50.decode_predictions(preds))
# print('Activations:', model.layers[0])
