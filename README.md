# DICOM Image Processing and Visualization

This Python script provides a series of functions for processing and visualizing DICOM (Digital Imaging and Communications in Medicine) medical imaging data. The script allows for loading and manipulating DICOM images, including converting them to Hounsfield Units (HU), applying lung window normalization, extracting metadata, reconstructing 2D aspects from slices, and creating GIF animations from DICOM series.

## Features

1. Load DICOM Slices: Read DICOM files from a directory and sort them based on their position in the patient's body.
2. DICOM to Hounsfield Units: Convert pixel values in DICOM slices to Hounsfield Units for standardized analysis.
3. Lung Window Normalization: Apply a lung window normalization to enhance the visibility of lung structures.
4. Extract DICOM Metadata: Extract and save selected DICOM metadata like 'StudyInstanceUID' and 'Modality' into a CSV file.
5. 2D Aspect Reconstruction: Reconstruct and visualize axial, sagittal, and coronal 2D aspects from DICOM slices.
6. Create GIF Animations: Compile a series of DICOM slices into a GIF animation, highlighting changes across slices.

## Installation

Ensure you have Python installed on your system. Additionally, install needed libraries listed in requirements.txt file:
```
pip3 install -r requirements.txt 
```

## Usage 
Here is an overview of how to use the functions in the script:
```
slice_visualization(slice_path, output_path)
metadata_to_csv(dir_path, output_path)   
slices_to_gif(dir_path, output_path)  
aspects_reconstruction(dir_path, output_path)
```
Replace dir_path and slice_path with the directory path where the dicom files or one chosen dicom file is. Replace output_file with output file name, according to function's output. 
Finally, execute:
```
python3 main.py 
```
