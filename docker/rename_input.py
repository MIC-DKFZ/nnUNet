import os

inputDir = os.getenv("INPUTDIR")
count = 0

# Nifti files are expected to be placed within INPUTDIR. 
# They will be renamed to match the naming pattern. 
print("Start renaming ...")
for filename in os.listdir(inputDir):
    old_path = os.path.join(inputDir, filename)

    if (os.path.isfile(old_path)):
        new_path = os.path.join(inputDir, "dcm{}_0000.nii.gz".format(count))
        print("Copy {0} to {1} ...".format(old_path, new_path))
        os.rename(old_path, new_path)
        count +=1

print("{} files processed".format(count))
print("...Renaming finished.")
