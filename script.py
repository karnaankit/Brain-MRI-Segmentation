import streamlit as st
import os
import numpy as np
import nibabel as nib
import shutil
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
 
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 25
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NON-ENHANCING', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}
model = tf.keras.models.load_model('dice_only.h5', compile=False)
 
 
def main():
    st.title("Brain MRI Segmentation App")
    if os.path.exists("data"):
        shutil.rmtree("data")
        os.makedirs("data")
    else:
        os.makedirs("data")
 
    uploaded_files = st.file_uploader("Choose nii files", type=["nii"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_name = file.name
            file_path = os.path.join("data", file_name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        case = file_name.split('_')[2]
    try:
        def predictByPath(case_path, case):
            X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
            vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
            flair = nib.load(vol_path).get_fdata()
 
            vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
            ce = nib.load(vol_path).get_fdata()
 
            for j in range(VOLUME_SLICES):
                X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            return model.predict(X / np.max(X), verbose=1)
 
        slice_no = st.slider('Select Slice', min_value=20, max_value=120, value= 50)
        def showPredictsById(case, start_slice=slice_no):
            path = 'D:/mri-seg/data'
            gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
            origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
            p = predictByPath(path, case)
            core = p[:, :, :, 1]
            edema = p[:, :, :, 2]
            enhancing = p[:, :, :, 3]
            plt.figure(figsize=(30,30))
            f, axarr = plt.subplots(6, 1, figsize=(30, 30))
            for i in range(6):  # for each image, add brain background
                axarr[i].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
                                cmap="gray", interpolation='none')
 
            axarr[0].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)),
                            cmap="gray")
            axarr[0].title.set_text('Original image flair')
            curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_NEAREST)
            axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
            axarr[1].title.set_text('Ground truth')
            axarr[2].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
            axarr[2].title.set_text('all classes')
            axarr[3].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
            axarr[4].imshow(core[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
            axarr[5].imshow(enhancing[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
            axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
            plt.savefig('segment.png', bbox_inches='tight')
            st.image('segment.png', use_column_width=False)
        showPredictsById(case)
 
    except:
        pass
 
 
if __name__ == '__main__':
    main()