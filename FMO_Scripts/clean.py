# Clean up the output files of the main script

import os

dir_name = "/Volumes/HDD/Research/PURDUE/SCRIPT_SUMMARY/combined_and_ready"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))
