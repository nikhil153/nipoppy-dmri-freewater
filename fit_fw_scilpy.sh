#!/bin/bash

# SUBJ=$1
# SESS=$2
# VERS=$3

# inputs
SUBJ=141135
SESS=BL
VERS=2.4.2

# fixed paths
DATADIR=/lustre06/project/6061841/bcmcpher/fw-dti
OUTVERS=0.9.0

# build derivative input directory path
TFDIR=$DATADIR/derivatives/tractoflow/$VERS/ses-$SESS/sub-$SUBJ

# build input file paths
INPDWI=$TFDIR/Resample_DWI/sub-${SUBJ}__dwi_resampled.nii.gz

# grab bval / bvec from either folder
if [ -d $TFDIR/Eddy_Topup ]; then
	INPBVAL=$TFDIR/Eddy_Topup/sub-${SUBJ}__bval_eddy
	INPBVEC=$TFDIR/Eddy_Topup/sub-${SUBJ}__dwi_eddy_corrected.bvec
elif [ -d $TFDIR/Eddy ]; then
	INPBVAL=$TFDIR/Eddy/sub-${SUBJ}__bval_eddy
	INPBVEC=$TFDIR/Eddy/sub-${SUBJ}__dwi_eddy_corrected.bvec
else
	echo "No valid bval / bvec files found."
	exit 1
fi

# output directory
OUTDPY=$DATADIR/derivatives/fwdti/$OUTVERS/ses-$SESS/sub-$SUBJ/dipy
OUTSPY=$DATADIR/derivatives/fwdti/$OUTVERS/ses-$SESS/sub-$SUBJ/scilpy

# output file stems
DPYNAM=$OUTDPY/sub-${SUBJ}_ses-${SESS}
SPYNAM=$OUTSPY/sub-${SUBJ}_ses-${SESS}_model-fwdti

# run dipy to create mask, regular tensor and fwdti (if supported by the data)
python fit_fw_dipy.py --dwi_data $INPDWI --dwi_bval $INPBVAL --dwi_bvec $INPBVEC --output_stem $DPYNAM

# run the amico fw model through scilpy
scil_compute_freewater.py $INPDWI $INPBVAL $INPBVEC --out_dir $OUTSPY

# rename scilpy outputs
mv $OUTSPY/dwi_fw_corrected.nii.gz ${SPYNAM}_desc-fwcorr_dwi.nii.gz
mv $OUTSPY/FIT_dir.nii.gz ${SPYNAM}_param-dir.nii.gz
mv $OUTSPY/FIT_FiberVolume.nii.gz ${SPYNAM}_param-fibervolume.nii.gz
mv $OUTSPY/FIT_FW.nii.gz ${SPYNAM}_param-freewater_map.nii.gz
mv $OUTSPY/FIT_nrmse.nii.gz ${SPYNAM}_param-nrmse_map.nii.gz

# copy bval/bvec files for fw corrected dwi data
cp $INBVAL ${SPYNAM}_desc-fwcorr_dwi.bval
cp $INBVEC ${SPYNAM}_desc-fwcorr_dwi.bvec

# create the fw tensor metric parameter map files
scil_compute_dti_metrics.py ${OUTSPY}_desc-fwcorr_dwi.nii.gz $INPBVAL $INPBVEC \
							--tensor ${OUTSPY}_param-tensor_map.nii.gz \
							--evals ${OUTSPY}_param-evals_map.nii.gz \
							--evecs ${OUTSPY}_param-evecs_map.nii.gz \
							--rgb ${OUTSPY}_param-rgb_map.nii.gz \
							--fa ${OUTSPY}_param-fa_map.nii.gz \
							--ga ${OUTSPY}_param-ga_map.nii.gz \
							--md ${OUTSPY}_param-md_map.nii.gz \
							--ad ${OUTSPY}_param-ad_map.nii.gz \
							--rd ${OUTSPY}_param-rd_map.nii.gz \
							--mode ${OUTSPY}_param-mode_map.nii.gz \
							--norm ${OUTSPY}_param-norm_map.nii.gz \
							--non-physical ${OUTSPY}_param-nonphysical_mask.nii.gz \
							--pulsation ${OUTSPY}_param-pulsation_map.nii.gz \
							--residual ${OUTSPY}_param-residual_map.nii.gz

# rename files for symmetry w/ dipy / better bids derivative compliance
mv ${SPYNAM}_param-evals_map_e1.nii.gz ${SPYNAM}_param-eval1_map.nii.gz
mv ${SPYNAM}_param-evals_map_e2.nii.gz ${SPYNAM}_param-eval2_map.nii.gz
mv ${SPYNAM}_param-evals_map_e3.nii.gz ${SPYNAM}_param-eval3_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v1.nii.gz ${SPYNAM}_param-evec1_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v2.nii.gz ${SPYNAM}_param-evec2_map.nii.gz
mv ${SPYNAM}_param-evecs_map_v3.nii.gz ${SPYNAM}_param-evec3_map.nii.gz
mv ${SPYNAM}_param-pulsation_map_std_dwi.nii.gz ${SPYNAM}_desc-dwi_pulsationSTD_map.nii.gz
mv ${SPYNAM}_param-pulsation_map_std_b0.nii.gz ${SPYNAM}_desc-b0_pulsationSTD_map.nii.gz
