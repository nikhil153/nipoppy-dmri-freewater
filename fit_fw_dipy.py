import argparse
import shutil
from warnings import warn

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table, check_multi_b
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import norm, geodesic_anisotropy
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti

OUTPUT_VERSION = "1.0.0"


def _saveNifti(data, affine, file_name):
    onii = nib.nifti1.Nifti1Image(data, affine)
    nib.save(onii, file_name)


def dpSaveTensorMetric(metric, affine, file_stem):
    try:
        out_data = eval(metric)
        _saveNifti(out_data, affine, file_stem)
        return(out_data)
    except Exception as e:
        warn(f"Creating the metric {metric} failed:\n\t{e}")
        return


def dipyEstimateResidualNRMSE(data, gtab, mask, dipyModel):

    import dipy

    # extract some data for model fitting
    dshape = data.shape
    b0s = data[..., gtab.b0s_mask]
    ab0 = np.mean(b0s, axis=-1, keepdims=True)
    # dwi = data[..., ~gtab.b0s_mask]

    # predicted signal
    prd = np.zeros(dshape, dtype="float32")

    # for each x-axis
    for i in range(dshape[0]):

        try:
            # fit and predict the whole data set w/ just that x-slice
            tmpfit = dipyModel.fit(data[i, :, :, :], mask[i, :, :])

            # get the predicted features based on model type
            if isinstance(dipyModel, dipy.reconst.dti.TensorModel):
                prd[i, :, :, :] = tmpfit.predict(gtab, np.squeeze(ab0[i, :, :]))
            elif isinstance(dipyModel, dipy.reconst.fwdti.FreeWaterTensorModel):
                prd[i, :, :, :] = tmpfit.predict(gtab)
            else:
                raise ValueError("Unsupported dipy.reconst model provided.")

        except StopIteration:
            # handle instances where input slice is fully masked as background voxels
            prd[i, :, :, :] = np.zeros(dshape[1:])

    # compute each directions residual / overall
    diff_data = np.abs(prd - data)

    # find the average residual estimate from orthogonal prediction of data
    res = np.mean(diff_data, axis=-1)

    # range: res / np.max(diff_data) - np.min(diff_data)
    # interquartile range: res / (iqr75 - irq25)
    # mean / coefficient of variation?: res / np.mean(dwi)
    # Mean Absolute Error? (MAE)

    # nrmse
    mse = ((prd - data) ** 2).mean(axis=-1)
    rmse = np.sqrt(mse)

    # max - min for _that_ nmrse correction
    mnmx = np.max(data, axis=-1) - np.min(data, axis=-1)  # range

    # possible nrmse corrections
    nrmse = rmse / mnmx  # nrmse
    # nrmse = rmse / np.mean(data, axis=-1)  # coefficient of variation?

    # set nan to 0
    nrmse[np.isnan(nrmse)] = 0

    return(res, nrmse)


def dpEstimatePulsationImplausible(data, gtab, affine, output_stem):

    # extract some data for model fitting
    b0s = data[..., gtab.b0s_mask]
    ab0 = np.mean(b0s, axis=-1, keepdims=True)
    dwi = data[..., ~gtab.b0s_mask]

    # non-physical / physically implausible signal
    pis = np.max(ab0 < dwi, axis=-1)
    _saveNifti(pis.astype("int16"), affine, f"{output_stem}desc-dwi_param-nonphysical_mask.nii.gz")

    # pulstation - dwi
    dwisd = np.std(data[..., ~gtab.b0s_mask], axis=-1)
    _saveNifti(dwisd, affine, f"{output_stem}desc-dwi_param-pulsationSTD_map.nii.gz")

    # pulsation - b0 - has to have more than 1 b0 to estimate
    if np.sum(gtab.b0s_mask) <= 1:
        print("Not enough b0 volumes to estimate b0 pulsation.")
        b0ssd = np.zeros(data.shape[:2]).astype("int16")
    else:
        if np.sum(gtab.b0s_mask) == 2:
            print(" -- WARNING - only 2 b0s to estimate pulsation - be careful with interpretation.")
        b0ssd = np.std(data[..., gtab.b0s_mask], axis=-1)

    _saveNifti(b0ssd, affine, f"{output_stem}desc-b0_param-pulsationSTD_map.nii.gz")

    # return signal plausibility and pulsation estimates
    return(pis, dwisd, b0ssd)


#
# configure parser
#

parser = argparse.ArgumentParser(
    prog="fit_fw_dipy",
    description="Fit the DTI and fwDTI model (if it's supported by the data) and produce tensor metric parameter maps with dipy.")

# add arguments
parser.add_argument("--dwi_data", type=str, required=True,
                    help="The preprocessed DWI dataset to estimate tensor model(s).")
parser.add_argument("--dwi_bval", type=str, required=True,
                    help="The bval file that corresponds to the --dwi_data input.")
parser.add_argument("--dwi_bvec", type=str, required=True,
                    help="The bvec file that corresponds to the --dwi_data input.")
parser.add_argument("--output_stem", type=str, required=True,
                    help="The file stem for result file names.")

print("Parsing input arguments...")

# parse and extract
args = parser.parse_args()
dwi_data = args.dwi_data
dwi_bval = args.dwi_bval
dwi_bvec = args.dwi_bvec
output_stem = args.output_stem

# what sanity checks are needed here?

#
# load the preprocessed imaging data out
#

print("Loading input data...")

# create np array of image data block
img = nib.load(dwi_data)
data = img.get_fdata()
affine = img.affine

# load the gradient table from the inputs
gtab = gradient_table(str(dwi_bval), str(dwi_bvec))

# create a mask based on the data
print(" -- Creating simple Median OTSU mask for tensor fitting...")
_, mask = median_otsu(data, median_radius=5, numpass=5,
                      vol_idx=[0, 1, 2], dilate=5)

# write the mask to disk for reuse
mout = nib.nifti1.Nifti1Image(mask.astype("int16"), affine)
nib.save(mout, f"{output_stem}_desc-brain_mask.nii.gz")

print(" -- Estimating Signal Plausibility and Pulsation...")
# estimate some basic signal properties of the input data
_ = dpEstimatePulsationImplausible(data, gtab, affine, f"{output_stem}_")

# fit regular tensor for comparison
print(" -- Fitting diffusion tensor model and estimating parameter maps...")

# initialize the regular tensor model
dtimodel = dti.TensorModel(gtab)

# estimate the regular tensor model
dtifit = dtimodel.fit(data, mask=mask)

# create output stem for dti file names
dt_out = f"{output_stem}_model-dti_param-%s_map.nii.gz"

# estimate residual / nrmse from regular tensor fit
dtres, dtnrmse = dipyEstimateResidualNRMSE(data, gtab, mask, dtimodel)
_saveNifti(dtres, affine, f"{output_stem}_model-dti_param-residual_map.nii.gz")
_saveNifti(dtnrmse, affine, f"{output_stem}_model-dti_param-nrmse_map.nii.gz")

# save regular tensor metrics from regular tensor
_ = dpSaveTensorMetric("dtifit.fa", affine, dt_out.replace("%s", "fa"))
_ = dpSaveTensorMetric("geodesic_anisotropy(dtifit.evals)", affine, dt_out.replace("%s", "ga"))
_ = dpSaveTensorMetric("dtifit.md", affine, dt_out.replace("%s", "md"))
_ = dpSaveTensorMetric("dtifit.rd", affine, dt_out.replace("%s", "ad"))
_ = dpSaveTensorMetric("dtifit.ad", affine, dt_out.replace("%s", "rd"))
_ = dpSaveTensorMetric("dtifit.mode", affine, dt_out.replace("%s", "mode"))
_ = dpSaveTensorMetric("dtifit.color_fa", affine, dt_out.replace("%s", "rgb"))
_ = dpSaveTensorMetric("dtifit.evals", affine, dt_out.replace("%s", "evals"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evals[:, :, :, 0])", affine, dt_out.replace("%s", "eval1"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evals[:, :, :, 1])", affine, dt_out.replace("%s", "eval2"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evals[:, :, :, 2])", affine, dt_out.replace("%s", "eval3"))
_ = dpSaveTensorMetric("dtifit.evecs", affine, dt_out.replace("%s", "evecs"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evecs[:, :, :, :, 0])", affine, dt_out.replace("%s", "evec1"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evecs[:, :, :, :, 1])", affine, dt_out.replace("%s", "evec2"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.evecs[:, :, :, :, 2])", affine, dt_out.replace("%s", "evec3"))
_ = dpSaveTensorMetric("dtifit.lower_triangular()", affine, dt_out.replace("%s", "tensor"))
_ = dpSaveTensorMetric("norm(dtifit.quadratic_form)", affine, dt_out.replace("%s", "norm"))
_ = dpSaveTensorMetric("np.squeeze(dtifit.directions)", affine, dt_out.replace("%s", "dir"))

# check condition of gtab if freewater tensor is even possible
if check_multi_b(gtab, 3, non_zero=False):

    print("Data has sufficient shells for Dipy fwDTI.")
    print(" -- Fitting freewater corrected diffusion tensor model and estimating parameter maps...")

    # initialize the freewater model
    fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

    # fit the freewater model
    fwdtifit = fwdtimodel.fit(data, mask=mask)

    # create outputstem for fwdti file names
    fw_out = f"{output_stem}_model-fwdti_param-%s_map.nii.gz"

    # get the fw corrected signal
    fw_pred = fwdtifit.predict(gtab)
    fw_corr = np.abs(fw_pred - data)
    _saveNifti(fw_corr, affine, f"{output_stem}_model-fwdti_desc-fwcorr_dwi.nii.gz")

    # copy the bval / bvec files for fw corrected dwi data
    shutil.copyfile(dwi_bval, f"{output_stem}_model-fwdti_desc-fwcorr_dwi.bval")
    shutil.copyfile(dwi_bvec, f"{output_stem}_model-fwdti_desc-fwcorr_dwi.bvec")

    # estimate residual / nrmse
    fwres, fwnrmse = dipyEstimateResidualNRMSE(data, gtab, mask, fwdtimodel)
    _saveNifti(fwres, affine, f"{output_stem}_model-fwdti_param-residual_map.nii.gz")
    _saveNifti(fwnrmse, affine, f"{output_stem}_model-fwdti_param-nrmse_map.nii.gz")

    # save regular tensor metrics from freewater tensor
    _ = dpSaveTensorMetric("fwdtifit.fa", affine, fw_out.replace("%s", "fa"))
    _ = dpSaveTensorMetric("geodesic_anisotropy(fwdtifit.evals)", affine, fw_out.replace("%s", "ga"))
    _ = dpSaveTensorMetric("fwdtifit.md", affine, fw_out.replace("%s", "md"))
    _ = dpSaveTensorMetric("fwdtifit.rd", affine, fw_out.replace("%s", "rd"))
    _ = dpSaveTensorMetric("fwdtifit.ad", affine, fw_out.replace("%s", "ad"))
    _ = dpSaveTensorMetric("fwdtifit.mode", affine, fw_out.replace("%s", "mode"))
    _ = dpSaveTensorMetric("fwdtifit.color_fa", affine, fw_out.replace("%s", "rgb"))
    _ = dpSaveTensorMetric("fwdtifit.evals", affine, fw_out.replace("%s", "evals"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evals[:, :, :, 0])", affine, fw_out.replace("%s", "eval1"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evals[:, :, :, 1])", affine, fw_out.replace("%s", "eval2"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evals[:, :, :, 2])", affine, fw_out.replace("%s", "eval3"))
    _ = dpSaveTensorMetric("fwdtifit.evecs", affine, fw_out.replace("%s", "evecs"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evecs[:, :, :, :, 0])", affine, fw_out.replace("%s", "evec1"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evecs[:, :, :, :, 1])", affine, fw_out.replace("%s", "evec2"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.evecs[:, :, :, :, 2])", affine, fw_out.replace("%s", "evec3"))
    _ = dpSaveTensorMetric("fwdtifit.lower_triangular()", affine, fw_out.replace("%s", "tensor"))
    _ = dpSaveTensorMetric("norm(fwdtifit.quadratic_form)", affine, fw_out.replace("%s", "norm"))
    _ = dpSaveTensorMetric("np.squeeze(fwdtifit.directions)", affine, fw_out.replace("%s", "dir"))

    # freewater tensor metrics
    dpSaveTensorMetric("fwdtifit.f", affine, fw_out.replace("%s", "freewater"))
    dpSaveTensorMetric("1 - fwdtifit.f", affine, fw_out.replace("%s", "fibervolume"))  # mask so background isn't 1

else:
    print("Data has insufficient b-values to fit dipy FreeWaterTensorModel.")

print("Done.")
