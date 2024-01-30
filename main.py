import numpy as np
import pandas as pd
import imageio
from IPython.display import Image
from pydicom import dcmread
from pathlib import Path
import os
import pydicom as dicom
import glob
from matplotlib import pyplot as plt


# functions definition 

def load_scan(path: str) -> list:
    """
    Load and process DICOM slices from a specified directory.
    This function reads DICOM files located in a given path, sorts them based on their position in the patient's body, and calculates the slice thickness. Each DICOM slice is updated with the computed slice thickness.
    Parameters:
    path (str): The file system path to the directory containing DICOM files.
    Returns:
    list: A sorted list of DICOM slices. Each slice is an object containing DICOM file data.
    """
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices: list) -> np.ndarray:
    """
    Convert DICOM slice pixel arrays to Hounsfield Units (HU).
    This function takes a list of DICOM slice objects, extracts their pixel arrays, and converts these pixel values into Hounsfield Units. The process involves adjusting the pixel values using the rescale intercept and slope as specified in the DICOM standard. It also handles pixels outside of the scan area by setting them to zero.
    Parameters:
    slices (list): A list of DICOM slice objects.
    Returns:
    npy.ndarray: A 3D numpy array where each 2D slice corresponds to a DICOM slice, with pixel values converted to Hounsfield Units.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # set outside-of-scan pixels to 0
    # the intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def set_lungwin(img: np.ndarray, hu=[-1200., 600.]) -> np.ndarray:
    """
    Apply a lung window normalization to a given image based on specified Hounsfield Unit (HU) values.
    This function normalizes the pixel values of a medical image (typically a CT scan) to enhance the visibility of lung structures. It linearly scales the Hounsfield Unit (HU) values to a specified window and then normalizes these values to a 0-255 scale, suitable for typical image display purposes.
    Parameters:
    img (np.ndarray): A 2D numpy array representing the image to be processed.
    hu (list, optional): A list of two float values representing the lower and upper bounds of the HU window. Default is [-1200.0, 600.0].
    Returns:
    np.ndarray: A 2D numpy array of the processed image, with pixel values normalized and scaled to 0-255 (uint8).
    """
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg

def extract_dicom_meta_data(path: str) -> dict:
    """
    Extract specific DICOM metadata from all DICOM files in a given directory.
    This function processes a directory containing DICOM files, extracting the 'StudyInstanceUID' and 'Modality' metadata from each file. 
    Parameters:
    path (str): The file system path to the directory containing DICOM files.
    Returns:
    dict: A dictionary with two keys: 'StudyInstanceUID' and 'Modality'. Each key maps to a list containing the respective metadata values from each DICOM file in the directory.
    """
    filename = load_scan(path)
    study_instance_uid_arr = []
    modality_arr = []
    for files in filename:
        StudyInstanceUID = files.StudyInstanceUID
        Modality = files.Modality
        study_instance_uid_arr.append(StudyInstanceUID)
        modality_arr.append(Modality)
    return {
        "StudyInstanceUID": study_instance_uid_arr,
        "Modality" : modality_arr}

def slice_visualization(slice_path: str, output_path: str) -> None:
        """
    Visualize a DICOM slice and save the image to output path.
    Parameters:
    slice_path (str): The file system path to the DICOM file to be visualized.
    Returns: None
    """
        slice = dcmread(slice_path)
        plt.imshow(slice.pixel_array, cmap=plt.cm.gray)
        plt.savefig(output_path)
        plt.show()

def metadata_to_csv(dir_path: str, csv_path: str) -> None:
    """
    Extract DICOM metadata from a directory and save it as a CSV file.
    Parameters:
    dir_path (str): The file system path to the directory containing DICOM files.
    csv_path (str): The file system path where the CSV file will be saved.
    Returns:
    None
    """
    meta_data = extract_dicom_meta_data(dir_path)
    df = pd.DataFrame(meta_data)
    df.to_csv(csv_path)
   

def slices_to_gif(dir_path: str, output_path: str) -> None:
    """
    Convert a series of DICOM slices to a GIF animation.
    This function reads all DICOM slices from a specified directory, applies lung window normalization and Hounsfield Unit (HU) conversion, and then compiles these processed images into a GIF animation. The GIF is saved to the specified output path.
    Parameters:
    dir_path (str): The file system path to the directory containing the DICOM slice files.
    output_path (str): The file system path where the resulting GIF should be saved.
    Returns:
    None
    """
    patient = load_scan(dir_path)
    patient_array = set_lungwin(get_pixels_hu(patient))
    imageio.mimsave(output_path, patient_array, duration=0.1)
    #Image(filename = output_path, format='png')


def aspects_reconstruction(dir_path: str, output_path: str) -> None:
    """
    Reconstruct and visualize 2D aspects (axial, sagittal, coronal) from a series of DICOM slices.
    This function loads DICOM slices from a given directory, calculates the appropriate aspect ratios for axial, sagittal, and coronal views based on the pixel spacing and slice thickness, and then constructs these views from the 3D image data. The reconstructed views are plotted and saved as a single image file.
    Parameters:
    dir_path (str): The file system path to the directory containing DICOM slice files.
    output_path (str): The file system path where the combined visualization image should be saved.
    Returns:
    None
    """
    patient = load_scan(dir_path)
    pixel_spacing = patient[0].PixelSpacing
    slice_thickness = patient[0].SliceThickness
    ax_aspect = pixel_spacing[1] / pixel_spacing[0]
    sag_aspect = pixel_spacing[1] / slice_thickness
    cor_aspect = slice_thickness / pixel_spacing[0]
    img_shape = list(patient[0].pixel_array.shape)
    img_shape.append(len(patient))
    img3d = np.zeros(img_shape)
    # fill 3D array with the images from the files
    for i, s in enumerate(patient):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    # plots 
    first = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, img_shape[2] // 2])
    first.set_aspect(ax_aspect)

    second = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, img_shape[1] // 2, :])
    second.set_aspect(sag_aspect)

    third = plt.subplot(2, 2, 3)
    plt.imshow(img3d[img_shape[0] // 2, :, :].T)
    third.set_aspect(cor_aspect)
    # save and show figures
    plt.savefig(output_path)
    plt.show()


if __name__ == '__main__':
    # path to the directory with all CT slices
    dir_path = "data"
    # path to the first dcm slice
    first_slice_path =  "data/1-001.dcm"
    # example of the visualisation of first slice
    slice_visualization(first_slice_path, output_path='output/first_slice.png')
    # extract the meta data: StudyInstanceUID and modality
    metadata_to_csv(dir_path, csv_path="output/metadata.csv")
    # make a gif of given slices and save them to the same directory
    slices_to_gif(dir_path, output_path="output/slices.gif")
    # reconstruction of patient : axial, sagittal and coronal aspects
    aspects_reconstruction(dir_path, output_path = "output/aspects_reconstruction.png")
    

    