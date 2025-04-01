#!/bin/bash

# add default usage for container check
if [ "$#" -lt 6 ]; then
	echo "$0: Estimate a DTI / fwDTI model with common parameter and residual estimation"
	echo ""
	echo "Usage: $0 <subject> <session> <dwi_file> <bval> <bvec> <output_dir> <use_shells>"
	echo ""
	echo "  <subject>     - subject ID - for name in output path"
	echo "  <session>     - session ID - for name in output path"
	echo "  <dwi_file>    - dMRI data file"
	echo "  <bval>        - corresponding bval file"
	echo "  <bvec>        - corresponding bvec file"
	echo "  <output_dir>  - output directory"
	echo "  <use_shells>  - subset to a specific set of shells (optional)"
	echo "                  bzero (0) shell is always required"
	echo '                  e.g. "0 1000" - double quoted, space separated'
	echo ""
	exit 1
fi

# input naming conventions
SUBJ=$1
SESS=$2

# input file names
INPDWIS=$3
INPBVAL=$4
INPBVEC=$5

# check if input files exist
if [ ! -f ${INPDWIS} | ! -f ${INPBVAL} | ! -f ${INPBVEC} ]; then
	echo "Input DWI file(s) not found:"
	echo "    DWI:  ${INPDWIS}"
	echo "    BVAL: ${INPBVAL}"
	echo "    BVEC: ${INPBVEC}"
	exit 1
fi

# input / output paths
OUTPATH=$6

# subset to a specific set of shells
USESHELL=$7

# output directory
OUTDPY=$OUTPATH/sub-${SUBJ}/ses-${SESS}/dipy
OUTSPY=$OUTPATH/sub-${SUBJ}/ses-${SESS}/scilpy

# create output directories
mkdir -p $OUTDPY
mkdir -p $OUTSPY

# output file stems
DPYNAM=$OUTDPY/sub-${SUBJ}_ses-${SESS}
SPYNAM=$OUTSPY/sub-${SUBJ}_ses-${SESS}_model-fwdti

# if USESHELL is set, subset the data to the specified shells
if [ -n "$USESHELL" ]; then

	# create a workdir to store subset shells
	echo "Creating working directory to store subset shells..."
	WORKDIR=${OUTPATH}/sub-${SUBJ}/ses-${SESS}/work
	mkdir -p $WORKDIR

	# fix shells to a not awful file name part
	SHELLS=b$(echo $USESHELL | tr -s "[:blank:]+" "b")

	# create new input stem
	WRKNAM=${WORKDIR}/sub-${SUBJ}_ses-${SESS}_acq-${SHELLS}_dwi

	# subset the data to the specified shells
	scil_extract_dwi_shell.py $INPDWIS $INPBVAL $INPBVEC $USESHELL \
							  ${WRKNAM}.nii.gz ${WRKNAM}.bval ${WRKNAM}.bvec \
							  -f --tolerance 15

	# update the input names
	INPDWIS=${WRKNAM}.nii.gz
	INPBVAL=${WRKNAM}.bval
	INPBVEC=${WRKNAM}.bvec

fi

# run dipy to create mask, regular tensor and fwdti (if supported by the data)
python /opt/fwdti/fit_fw_dipy.py --dwi_data $INPDWIS --dwi_bval $INPBVAL --dwi_bvec $INPBVEC --output_stem $DPYNAM

# run the amico fw model through scilpy
scil_compute_freewater.py $INPDWIS $INPBVAL $INPBVEC --out_dir $OUTSPY -f

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

# remove the .npy files, .png and config.pickle
rm ${OUTSPY}/*.npy ${OUTSPY}/*.png ${OUTSPY}/config.pickle
