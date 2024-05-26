import zipfile
import pathlib

current_path = pathlib.Path(__file__).parent.absolute()

# make path for folder name: all_data

all_data_path = current_path / 'all_data'



zip_file_path = 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
        
    # Calculate the total number of files to extract
    total_files = len(file_list)
    
    for index, file_name in enumerate(file_list, start=1):
        # Calculate the progress percentage
        progress = (index / total_files) * 100
        
        # Print the progress
        print(f'Extracting {file_name} ({progress:.2f}% complete)')
        
        # Extract the file
        zip_ref.extract(file_name, all_data_path)
