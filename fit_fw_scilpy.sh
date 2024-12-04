#!/bin/bash

SUBJ=$1
SESS=$2
VERS=$3

INPDWI=$4
INPBVAL=$5
INPBVEC=$6

# inputs
#SUBJ=141135
#SESS=BL
#VERS=2.4.2

# fixed paths
# DATADIR=/lustre06/project/6061841/bcmcpher/fw-dti
#DATADIR=/data/origami/nikhil/datasets/sandbox/qpn
#OUTSDIR=/data/origami/bcmcpher/fwdti/results
DATADIR=/data/pd/ppmi-new
OUTSDIR=/data/pd/scratch
OUTVERS=0.9.0

# build derivative input directory path
TFDIR=$DATADIR/derivatives/tractoflow/v$VERS/output/ses-$SESS/sub-$SUBJ

# # build input file paths
# INPDWI=$TFDIR/Resample_DWI/sub-${SUBJ}__dwi_resampled.nii.gz

# # grab bval / bvec from either folder
# if [ -d $TFDIR/Eddy_Topup ]; then
# 	INPBVAL=$TFDIR/Eddy_Topup/sub-${SUBJ}__bval_eddy
# 	INPBVEC=$TFDIR/Eddy_Topup/sub-${SUBJ}__dwi_eddy_corrected.bvec
# 	echo "Found bval / bvec data."
# elif [ -d $TFDIR/Eddy ]; then
# 	INPBVAL=$TFDIR/Eddy/sub-${SUBJ}__bval_eddy
# 	INPBVEC=$TFDIR/Eddy/sub-${SUBJ}__dwi_eddy_corrected.bvec
# 	echo "Found bval / bvec data."
# else
# 	echo "No valid bval / bvec files found."
# 	exit 1
# fi

# output directory
OUTDPY=$OUTSDIR/fwdti/$OUTVERS/ses-$SESS/sub-$SUBJ/dipy
OUTSPY=$OUTSDIR/fwdti/$OUTVERS/ses-$SESS/sub-$SUBJ/scilpy

# create output directories
mkdir -p $OUTDPY
mkdir -p $OUTSPY

# output file stems
DPYNAM=$OUTDPY/sub-${SUBJ}_ses-${SESS}
SPYNAM=$OUTSPY/sub-${SUBJ}_ses-${SESS}_model-fwdti

# run dipy to create mask, regular tensor and fwdti (if supported by the data)
python /data/origami/bcmcpher/fwdti/bin/fit_fw_dipy.py --dwi_data $INPDWI --dwi_bval $INPBVAL --dwi_bvec $INPBVEC --output_stem $DPYNAM

# run the amico fw model through scilpy
scil_compute_freewater.py $INPDWI $INPBVAL $INPBVEC --out_dir $OUTSPY -f

# rename scilpy outputs
mv $OUTSPY/dwi_fw_corrected.nii.gz ${SPYNAM}_desc-fwcorr_dwi.nii.gz
mv $OUTSPY/FIT_dir.nii.gz ${SPYNAM}_param-dir.nii.gz
mv $OUTSPY/FIT_FiberVolume.nii.gz ${SPYNAM}_param-fibervolume.nii.gz
mv $OUTSPY/FIT_FW.nii.gz ${SPYNAM}_param-freewater_map.nii.gz
mv $OUTSPY/FIT_nrmse.nii.gz ${SPYNAM}_param-nrmse_map.nii.gz

# copy bval/bvec files for fw corrected dwi data
cp $INPBVAL ${SPYNAM}_desc-fwcorr_dwi.bval
cp $INPBVEC ${SPYNAM}_desc-fwcorr_dwi.bvec

# create the fw tensor metric parameter map files
scil_compute_dti_metrics.py ${SPYNAM}_desc-fwcorr_dwi.nii.gz $INPBVAL $INPBVEC \
                                                        --mask ${DPYNAM}_desc-brain_mask.nii.gz \
							--tensor ${SPYNAM}_param-tensor_map.nii.gz \
							--evals ${SPYNAM}_param-evals_map.nii.gz \
							--evecs ${SPYNAM}_param-evecs_map.nii.gz \
							--rgb ${SPYNAM}_param-rgb_map.nii.gz \
							--fa ${SPYNAM}_param-fa_map.nii.gz \
							--ga ${SPYNAM}_param-ga_map.nii.gz \
							--md ${SPYNAM}_param-md_map.nii.gz \
							--ad ${SPYNAM}_param-ad_map.nii.gz \
							--rd ${SPYNAM}_param-rd_map.nii.gz \
							--mode ${SPYNAM}_param-mode_map.nii.gz \
							--norm ${SPYNAM}_param-norm_map.nii.gz \
							--non-physical ${SPYNAM}_param-nonphysical_mask.nii.gz \
							--pulsation ${SPYNAM}_param-pulsation_map.nii.gz \
							--residual ${SPYNAM}_param-residual_map.nii.gz \
							--tensor_format dipy -f

# rename files for symmetry w/ dipy / better bids derivative compliance
mv ${SPYNAM}_param-evals_map_e1.nii.gz ${SPYNAM}_param-eval1_map.nii.gz
mv ${SPYNAM}_param-evals_map_e2.nii.gz ${SPYNAM}_param-eval2_map.nii.gz
mv ${SPYNAM}_param-evals_map_e3.nii.gz ${SPYNAM}_param-eval3_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v1.nii.gz ${SPYNAM}_param-evec1_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v2.nii.gz ${SPYNAM}_param-evec2_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v3.nii.gz ${SPYNAM}_param-evec3_map.nii.gz
mv ${SPYNAM}_param-pulsation_map_std_dwi.nii.gz ${SPYNAM}_desc-dwi_pulsationSTD_map.nii.gz
mv ${SPYNAM}_param-pulsation_map_std_b0.nii.gz ${SPYNAM}_desc-b0_pulsationSTD_map.nii.gz

# clean out the *.npy files and the .png plot?
# rm $OUTSPY/*.npy $OUTSPY/*.png
