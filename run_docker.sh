#!/bash/bin

docker run -it --rm -v <bids_dir>:/data:ro -v <output_dir>:/out fwdti 10101 BL /data/test-dwi.nii.gz /data/test-dwi.bval /data/test-dwi.bvec /out

# docker run -it --rm -v <bids_dir>:/data:ro -v <output_dir>:/out nipreps/mriqc:latest /data /out participant --participant_label 001 002 003
# docker run -it --rm -v <bids_dir>:/data:ro -v <output_dir>:/out nipreps/mriqc:latest /data /out group
