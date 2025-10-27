import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import dice

import json
import nibabel as nib
import logging

from dipy.align import affine_registration
from dipy.align import affine_registration, syn_registration, write_mapping, read_mapping
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.imaffine import AffineMap
from dipy.io.image import load_nifti
from dipy.viz import regtools

# import disptools.displacements as displacements 
import SimpleITK as sitk


# configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def jacobian_to_volume_change(jacobian: sitk.Image, epsilon: float = 1e-5) -> sitk.Image:
    r""" Convert a Jacobian map to a volume change map (source: https://github.com/m-pilia/disptools copied here to avoid dependancy).

    A volume change map is defined as

    .. math::
        VC[f](x) =
        \begin{cases}
            1 - \frac{1}{J[f](x)}  \quad &J[f](x) \in (0,1) \\
            J[f](x) - 1            \quad &J[f](x) \ge 1
        \end{cases}

    Parameters
    ----------
    jacobian : sitk.Image
        Input Jacobian map.
    epsilon : float
        Lower threshold for the Jacobian; any value below
        `epsilon` will be replaced with `epsilon`.

    Returns
    -------
    sitk.Image
        Volume change map associated to the input Jacobian.
    """

    data = sitk.GetArrayFromImage(jacobian)
    processed = np.empty(data.shape, dtype=data.dtype)

    ind_expa = data >= 1.0
    ind_comp = data < 1.0
    ind_sing = data <= epsilon

    data[ind_sing] = epsilon

    processed[ind_expa] = data[ind_expa] - 1.0
    processed[ind_comp] = 1.0 - (1.0 / data[ind_comp])

    result = sitk.GetImageFromArray(processed)
    # result.CopyInformation(jacobian) # Not needed for volume calculations
    return result   

def align_and_resample(mov, ref, label=None, affine_config=None, syn_config=None, sub2ref_affine=None, sub2ref_mapping=None, warp=True):
    """
    Registration (linear + non-linear) of moving image to reference image and resampling of labels in the reference space.
    Wrapping this into a single function for avoiding messy transform calls.
    Reference: https://docs.dipy.org/stable/examples_built/registration/streamline_registration.html

    Parameters
    ----------
    mov : nibabel.Nifti1Image
        The moving image to be registered and resampled.
    ref : nibabel.Nifti1Image
        The static reference image.
    label : nibabel.Nifti1Image, optional
        The label image in the reference space to be resampled, by default None.
    sub2ref_affine : np.ndarray, optional
        Precomputed affine transformation matrix from moving to static image space, by default None.
    sub2ref_mapping : DiffeomorphicMap, optional
        Precomputed diffeomorphic mapping from moving to reference image space, by default None.
    warp : bool, optional
        Whether to perform non-linear warping after affine registration, by default True.

    Returns
    -------
    sub_transformed : nibabel.Nifti1Image
        The registered and resampled moving image.
    ref_transformed : nibabel.Nifti1Image
        The registered and resampled static image.
    label_transformed : nibabel.Nifti1Image
        The resampled label image.
    sub2ref_affine : np.ndarray
        The affine transformation matrix from moving to static image space.
    sub2_ref_warp : DiffeomorphicMap
        The diffeomorphic mapping from moving to reference image space.

    """

    if sub2ref_affine is not None:
        logger.info(" --  --  -- Using provided affine for registration.")

    else:
        logger.info(" --  --  -- Computing affine registration.")
        _, sub2ref_affine = affine_registration(
            mov,
            ref,
            moving_affine=mov.affine,
            static_affine=ref.affine,
            nbins=affine_config.get("nbins", 32),
            metric=affine_config.get("metric", "MI"),
            pipeline=affine_config.get("pipeline", ["center_of_mass", "translation", "rigid", "affine"]),
            level_iters=affine_config.get("level_iters", [10000, 1000, 100]),
            sigmas=affine_config.get("sigmas",[3.0, 1.0, 0.0]),
            factors=affine_config.get("factors", [4, 2, 1]),
        )

    # Transform the moving image
    affmap = AffineMap(
        sub2ref_affine,
        domain_grid_shape=ref.shape,
        domain_grid2world=ref.affine,
        codomain_grid_shape=mov.shape,
        codomain_grid2world=mov.affine,
    )

    # apply the transformation to the moving image data
    sub_affined_data = affmap.transform(mov.get_fdata())
    sub_affined = nib.Nifti1Image(sub_affined_data, ref.affine)

    # compute non-linear warp if specified
    if warp:
        if sub2ref_mapping is not None:
            logger.info(" --  --  -- Using provided warp for registration.")

        else:
            logger.info(" --  --  -- Computing non-linear registration.")
            # still using the orginal moving image but with affine pre-alignment
            _, sub2ref_mapping = syn_registration(
                mov,
                ref,
                moving_affine=mov.affine,
                static_affine=ref.affine,
                prealign=sub2ref_affine,
            )

        sub_warped_data = sub2ref_mapping.transform(mov.get_fdata())
        sub_warped = nib.Nifti1Image(sub_warped_data, ref.affine)
    
    else:
        sub_warped = None
        sub2ref_mapping = None

    # inverse align the label image to the subject space and resample
    if label is not None:
        label_inverse_affined_data = affmap.transform_inverse(label.get_fdata(), interpolation='nearest')

        if warp:          
            label_inverse_warped_data = sub2ref_mapping.transform_inverse(label.get_fdata(), interpolation='nearest')  
            label_inverse = nib.Nifti1Image(label_inverse_warped_data, mov.affine) 
            
        else:
            label_inverse = nib.Nifti1Image(label_inverse_affined_data, mov.affine) 

        # Sanity check: apply the forward transform to see if we get back the original label           
        label_affine_looped_data = affmap.transform(label_inverse_affined_data, interpolation='nearest')

        # check overlap
        label_binary = (label.get_fdata() > 0).astype(int)        
        label_affine_looped_binary = (label_affine_looped_data > 0).astype(int)

        dice_label_aff_loop = 1 - dice(label_binary.flatten(), label_affine_looped_binary.flatten())

        logger.info(f" --  --  -- dice_label_aff_loop overlap (should be close to 1.0): {dice_label_aff_loop:.3f}")

        dice_score = {"label_aff_loop": dice_label_aff_loop}
        if warp:
            label_warp_looped_data = sub2ref_mapping.transform(label_inverse_warped_data, interpolation='nearest')
            label_warp_looped_binary = (label_warp_looped_data > 0).astype(int)            
            dice_label_warp_loop = 1 - dice(label_binary.flatten(), label_warp_looped_binary.flatten())
            logger.info(f" --  --  -- dice_label_warp_loop overlap (should be close to 1.0): {dice_label_warp_loop:.3f}")

            # compare inverse affine vs inverse warp
            label_inverse_affined_binary = (label_inverse_affined_data > 0).astype(int)
            label_inverse_warped_binary = (label_inverse_warped_data > 0).astype(int)
            dice_label_aff_warp_inverse = 1 - dice(label_inverse_affined_binary.flatten(), label_inverse_warped_binary.flatten())

            logger.info(f" --  --  -- dice_label_aff_warp_inverse overlap (should be between 0.7 - {dice_label_aff_loop:.2f}): {dice_label_aff_warp_inverse:.3f}")

            dice_score["label_warp_loop"] = dice_label_warp_loop
            dice_score["aff_warp_inverse"] = dice_label_aff_warp_inverse
        
            # mixed transform loops
            label_inverse_warp_affine_looped_data = affmap.transform(label_inverse_warped_data, interpolation='nearest')
            label_inverse_affine_warp_looped_data = sub2ref_mapping.transform(label_inverse_affined_data, interpolation='nearest')
            label_inverse_warp_affine_looped_binary = (label_inverse_warp_affine_looped_data > 0).astype(int)
            label_inverse_affine_warp_looped_binary = (label_inverse_affine_warp_looped_data > 0).astype(int)

            dice_label_inverse_warp_affine_loop = 1 - dice(label_binary.flatten(), label_inverse_warp_affine_looped_binary.flatten())
            logger.info(f" --  --  -- dice_label_warp_affine_loop overlap (should be between 0.7 - {dice_label_aff_loop:.2f}): {dice_label_inverse_warp_affine_loop:.3f}")

            dice_label_inverse_affine_warp_loop = 1 - dice(label_binary.flatten(), label_inverse_affine_warp_looped_binary.flatten())
            logger.info(f" --  --  -- dice_label_affine_warp_loop overlap (should be between 0.7 - {dice_label_aff_loop:.2f}): {dice_label_inverse_affine_warp_loop:.3f}")

            dice_score["label_inverse_warp_affine_loop"] = dice_label_inverse_warp_affine_loop
            dice_score["label_inverse_affine_warp_loop"] = dice_label_inverse_affine_warp_loop

            # Calculate total jacobian determinant statistics
            
            # generate simpleitk images for the displacement field
            sitk_disp = sitk.GetImageFromArray(sub2ref_mapping.forward.astype(np.float32))
            jacobian_det = sitk.DisplacementFieldJacobianDeterminant(sitk_disp)
            # vol_change = displacements.jacobian_to_volume_change(jacobian_det)
            vol_change = jacobian_to_volume_change(jacobian_det)
            mean_jacobian = np.mean(vol_change)
            std_jacobian = np.std(vol_change)
            total_jacobian = np.sum(vol_change)
            logger.info(f" --  --  -- Jacobian vol change stats - mean: {mean_jacobian:.3f}, std: {std_jacobian:.3f}, total: {total_jacobian:.3f}")

            jacobian_stats = {
                "mean": mean_jacobian,
                "std": std_jacobian,
                "total": total_jacobian
            }

            qc_metrics = {"dice_scores": dice_score, "jacobian_stats": jacobian_stats }
            
    else:
        label_inverse = None

    return sub_affined, sub_warped, label_inverse, sub2ref_affine, sub2ref_mapping, qc_metrics

def loadLabels(f_label_vol, f_label_map):

    # load the image
    label_vol = nib.load(f_label_vol)

    # get the unique labels
    labels_in_vol = np.unique(label_vol.get_fdata())[1:].astype(int)  # skip the background label

    logger.info(f" -- Found {len(labels_in_vol)} labels in the volume: {f_label_vol}")
 
    # read label map file
    logger.info(f" -- Loading label map file: {f_label_map}")
    label_map_df = pd.read_csv(f_label_map, sep="\t")
    
    # check map file columns
    if "label" not in label_map_df.columns:
        raise ValueError("Text labels file does not contain 'label' column.")   
    if "name" not in label_map_df.columns:
        raise ValueError("Text labels file does not contain 'name' column.")
    
    labels_in_map = label_map_df["label"].to_list()
    logger.info(f" -- Found {len(labels_in_map)} labels in the text file: {f_label_map}")
     
    # check if all the labels in the volume are in the text file
    if set(labels_in_vol).issubset(set(labels_in_map)):
        logger.info(" -- All volume labels are present in the text label file.")
    else:
        logger.error(" -- Some volume labels are not present in the text label file.")
        raise ValueError("Not all volume labels are present in the text label file.")

    # pull the names from the text file
    roi_labels = label_map_df["name"].to_list()
    
    # sort roi_labels by label number
    roi_labels = [x for _, x in sorted(zip(labels_in_map, roi_labels))]

    return(label_vol, labels_in_map, roi_labels)


def load_nii(nii_path, loader="nib"):
    """ Load a nifti file either using nibabel or dipy """

    try:
        if loader == "dipy":
            vol, affine = load_nifti(nii_path)
        else:
            nii = nib.load(nii_path)
            vol = nii.get_fdata()
            affine = nii.affine
    
    except Exception as e:
        logger.error(f" -- Error loading nifti file: {nii_path}")
        logger.exception(e)
        vol = None
        affine = None

    return vol, affine


def getROIStats(data, label_mask, roi_label):
    """ Get statistics for a given ROI """

    roi_stats = {}
    roi_data = data[label_mask == roi_label]

    roi_stats["mean"] = np.nanmean(roi_data)
    roi_stats["std"] = np.nanstd(roi_data)
    roi_stats["count"] = np.sum(~np.isnan(roi_data))

    return roi_stats

# argparse setup
parser = argparse.ArgumentParser(description="extract dMRI IDP stats from dipy fwDTI outputs.")
parser.add_argument("-d", "--dataset", help='Derivatives directory contained processed dMRI data', required=True)
parser.add_argument("-p", "--participant_id", help="participant ID", required=True)
parser.add_argument("-s", "--session_id", help="participant session", default="01")
parser.add_argument("-c", "--config_file", help="configuration file for registration parameters", required=True)
parser.add_argument("-a", "--affine_only", help="use only affine registration", action="store_true")
parser.add_argument("-t", "--test_reg_images", help="save test registration images", action="store_true")

args = parser.parse_args()

dataset = args.dataset 
participant_id = args.participant_id
session_id = args.session_id
config_file = args.config_file
affine_only = args.affine_only
save_test_reg_images = args.test_reg_images

# check if participant id has sub- prefix
bids_participant = participant_id #"sub-YLOPD31"
if not bids_participant.startswith("sub-"):
    bids_participant = f"sub-{bids_participant}"

# check is session id has sub- prefix
if session_id.startswith("ses-"):
    session_id = session_id.replace("ses-", "")
    
session = f"ses-{session_id}"

# read the config file
logger.info(f"Loading configuration file: {config_file}")
with open(config_file, 'r') as cf:
    config = json.load(cf)
 
 # preproc config
preproc_pipeline_name = config.get("preproc_pipeline").get("name")
preproc_pipeline_version = config.get("preproc_pipeline").get("version")
preproc_pipeline_software = config.get("preproc_pipeline").get("software")

logger.info(f"Using preproc pipeline: {preproc_pipeline_name} - version: {preproc_pipeline_version}")

# reference space config
ref_img = config.get("reference_space").get("image_path")
ref_label = config.get("reference_space").get("label_path")
label_map = config.get("reference_space").get("label_desc_path")
atlas_name = config.get("reference_space").get("name")

# registration config
affine_registration_params = config.get("affine_registration_params")

# diffusion metric config
diffusion_metrics = config.get("diffusion_metrics")

# setup input / output paths
pipeline_output_dir = f"{dataset}/derivatives/{preproc_pipeline_name}/{preproc_pipeline_version}/output/"
pipeline_idps_dir = f"{dataset}/derivatives/{preproc_pipeline_name}/{preproc_pipeline_version}/idp/"

# create output dir if not exists
Path(pipeline_idps_dir).mkdir(parents=True, exist_ok=True)

if save_test_reg_images:    
    test_reg_images_dir = "./test_reg_images"
    Path(test_reg_images_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving registration QC images for testing at {test_reg_images_dir} for sanity checks.")
 
logger.info("Running extraction of estimated features.")

# load label volume
logger.info("Loading and verifying label volume.")

# check and load the label image
labs, labels, roi_labels = loadLabels(ref_label, label_map)
parc = Path(ref_label).name.replace(".nii.gz", "")
nlabs = len(roi_labels)

label_map_df = pd.DataFrame()
label_map_df["roi_idx"] = labels
label_map_df["roi_name"] = roi_labels

# load the reference volume for coregistration
ref = nib.load(ref_img)

if affine_only:
    reg_method = "affine" 
else:
    reg_method = "affine+syn"

warp = not affine_only

logger.info(f"Using registration method: {reg_method} --> warp={warp}")

# build input / output paths
datadir = pipeline_output_dir
outsdir = pipeline_idps_dir

# for every subject
logger.info(f" -- Extracting IDP data from: {preproc_pipeline_name}-{preproc_pipeline_version}")

logger.info(f" --  -- Processing: {bids_participant}")

# create output file name
outfile = Path(outsdir, f"{bids_participant}_{session}_{preproc_pipeline_name}-{preproc_pipeline_version}_{parc}_{reg_method}_idp.tsv")

# load the files to extract
logger.info(f" --  -- Extracting data from: {bids_participant}")
dpdir = Path(datadir, bids_participant, f"{session}", "dipy")

mov = nib.load(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-fa_map.nii.gz"))

# save original images for sanity checks
if save_test_reg_images:
    nib.save(mov, f'{test_reg_images_dir}/sub_orig.nii.gz')
    nib.save(ref, f'{test_reg_images_dir}/ref_orig.nii.gz')
    nib.save(labs, f'{test_reg_images_dir}/label_orig.nii.gz')
    
logger.info(" --  --  -- Starting alignment and resampling.")  

# path to subject affine file
sub_aff_dir = f"{pipeline_output_dir}/{bids_participant}/{session}/affine/"
subj_aff_stem = f"{bids_participant}_{session}_{atlas_name}_sub2ref.txt"
subj_aff_path = Path(sub_aff_dir, subj_aff_stem)

# path to subject warp file
sub_warp_dir = f"{pipeline_output_dir}/{bids_participant}/{session}/warp/"
subj_warp_stem = f"{bids_participant}_{session}_{atlas_name}_sub2ref_mapping.nii.gz"
subj_warp_path = Path(sub_warp_dir, subj_warp_stem)

# if the affine file exists, load it
# load the warp ONLY if affine exists since we don't know the prealign otherwise
sub2ref_affine = None
sub2ref_mapping = None
if subj_aff_path.exists():
    logger.info(f" --  --  -- Using existing affine: {subj_aff_stem}")
    sub2ref_affine = np.loadtxt(subj_aff_path)

    # load existing warp if exists
    if warp & (subj_warp_path.exists()):
        logger.info(f"Using existing warp from: {subj_warp_path}")        
        # Need to specify prealign as affine since it was used during syn computation
        # For some reason it requires the inverse affine while reading the mapping
        # see: https://github.com/dipy/dipy/discussions/3272
        # sub2ref_mapping = read_mapping(str(subj_warp_path), mov, ref)
        sub2ref_mapping = read_mapping(str(subj_warp_path), mov, ref, prealign=np.linalg.inv(sub2ref_affine))

# align and resample the moving image to the reference
sub_affined, sub_warped, label_inverse, sub2ref_affine, sub2ref_mapping, qc_metrics = align_and_resample(
    mov, 
    ref, 
    labs, 
    affine_config=affine_registration_params,
    syn_config=None, 
    sub2ref_affine=sub2ref_affine, 
    sub2ref_mapping=sub2ref_mapping, 
    warp=warp
    )

logger.debug(f"shape of original images: mov: {mov.shape}, ref: {ref.shape}, label: {labs.shape}")

if warp:
    logger.debug(f"Shape of the sub transforms: sub_affined: {sub_affined.shape}, sub_warped: {sub_warped.shape}")
    logger.debug(f"shape of warps: forward: {sub2ref_mapping.forward.shape}, inverse: {sub2ref_mapping.backward.shape}")
else:
    logger.debug(f"Shape of the sub transforms: sub_affined: {sub_affined.shape}")

logger.debug(f"shape of label resampled images: inverse_affined: {label_inverse.shape}")

logger.info(" --  --  -- Alignment and resampling complete.")

# save the affine to disk
if not subj_aff_path.exists():
    Path(sub_aff_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f" --  --  -- Saving affine to: {subj_aff_path}")
    np.savetxt(subj_aff_path, sub2ref_affine)
    
# save warps
if warp & (not subj_warp_path.exists()):
    Path(sub_warp_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving warp to: {subj_warp_path}")
    write_mapping(sub2ref_mapping, subj_warp_path)

# save the transformed images
if save_test_reg_images:
    nib.save(sub_affined, f'{test_reg_images_dir}/sub_affined.nii.gz')
    if warp:
        nib.save(sub_warped, f'{test_reg_images_dir}/sub_warped.nii.gz')
        nib.save(label_inverse, f'{test_reg_images_dir}/label_inverse_warped.nii.gz')
    else:
        nib.save(label_inverse, f'{test_reg_images_dir}/label_inverse_affined.nii.gz')
    
# get resampled labels
if warp:
    tldat = label_inverse.get_fdata().astype(int)
else:
    tldat = label_inverse.get_fdata().astype(int)

# load and prep data for extraction
# diff_params = ["fa","md"]
diff_params = diffusion_metrics.keys()
logger.info(f" --  --  -- Extracting diffusion metrics: {diff_params}")
 
# load data files
logger.info(" --  --  -- Loading data files for extraction.")   
summary_df = pd.DataFrame() 
for diff_p in diff_params:
    diff_param_map, _ = load_nii(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-{diff_p}_map.nii.gz"), loader="dipy")
    diff_FW_param_map, _ = load_nii(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-{diff_p}_map.nii.gz"), loader="dipy")

    # for every roi label, get the mean value w/in the labels
    dti_roi_df = pd.DataFrame(columns=["roi_idx","model","param","mean_value","std_value","voxel_count"])
    fwdti_roi_df = pd.DataFrame(columns=["roi_idx","model","param","mean_value","std_value","voxel_count"])
    for idx, roi in enumerate(labels):
        # Default
        roi_stats = getROIStats(diff_param_map, tldat, roi)
        dti_val_mean, dti_val_std, dti_val_count = roi_stats["mean"], roi_stats["std"], roi_stats["count"]
        dti_roi_df.loc[idx] = [roi, "dti", diff_p, dti_val_mean, dti_val_std, dti_val_count]
        # FW
        roi_stats = getROIStats(diff_FW_param_map, tldat, roi)
        FW_val_mean, FW_val_std, FW_val_count = roi_stats["mean"], roi_stats["std"], roi_stats["count"]
        fwdti_roi_df.loc[idx] = [roi, "fwdti", diff_p, FW_val_mean, FW_val_std, FW_val_count]

    roi_df = pd.concat([dti_roi_df,fwdti_roi_df], axis=0)
    summary_df = pd.concat([summary_df, roi_df])

summary_df["participant_id"] = bids_participant
summary_df["pipeline"] = f"{preproc_pipeline_name}-{preproc_pipeline_version}"
summary_df["software"] = preproc_pipeline_software
summary_df["registration"] = reg_method

summary_df = pd.merge(summary_df,label_map_df,on="roi_idx",how="left")

# write the dataframe to disk
summary_df.to_csv(outfile, index=False, header=True, sep="\t")
logger.info(f" --  -- Saved {bids_participant}_{session} IDPs to {outfile}.")

# Save qc_metrics to the dataframe 
qc_df = pd.DataFrame()
dice_score = qc_metrics.get("dice_scores", {})
jacobian_stats = qc_metrics.get("jacobian_stats", {})

qc_df["bids_participant_id"] = [bids_participant]
qc_df["session"] = [session]
qc_df["pipeline"] = [f"{preproc_pipeline_name}-{preproc_pipeline_version}"]
qc_df["software"] = [preproc_pipeline_software]
qc_df["registration"] = [reg_method]    

for key, value in dice_score.items():
    qc_df[f"dice_{key}"] = value

for key, value in jacobian_stats.items():
    qc_df[f"Jacobian_{key}"] = value

qc_outfile = Path(outsdir, f"{bids_participant}_{session}_{preproc_pipeline_name}-{preproc_pipeline_version}_{parc}_{reg_method}_qc-metrics.tsv")
qc_df.to_csv(qc_outfile, index=False, header=True, sep="\t")
logger.info(f" --  -- Saved {bids_participant}_{session} QC metrics to {qc_outfile}.")

logger.info("IDP extraction complete!!")
