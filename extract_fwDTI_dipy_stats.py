import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import json
import nibabel as nib

from dipy.align import affine_registration
from dipy.align import affine_registration, syn_registration, write_mapping, read_mapping
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.imaffine import AffineMap
from dipy.io.image import load_nifti
from dipy.viz import regtools


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
        print(" --  --  -- Using provided affine for registration.")

    else:
        print(" --  --  -- Computing affine registration.")
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
        print(" --  --  -- Computing non-linear warp registration.")
        if sub2ref_mapping is not None:
            print(" --  --  -- Using provided warp for registration.")

        else:
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
        if warp:         
            # first, transform the ref label data using the diffeomorphic map   
            label_inverse_warped_data = sub2ref_mapping.transform_inverse(label.get_fdata(), interpolation='nearest')  
            label_inverse_warped = nib.Nifti1Image(label_inverse_warped_data, mov.affine)

            # then, apply the inverse affine transform
            label_inverse_affined_data = affmap.transform_inverse(label_inverse_warped_data, interpolation='nearest')          
        else:
            label_inverse_warped = None
            label_inverse_affined_data = affmap.transform_inverse(label.get_fdata(), interpolation='nearest')

        label_inverse_affined = nib.Nifti1Image(label_inverse_affined_data, mov.affine) 

    else:
        label_inverse_affined = None
        label_inverse_warped = None

    return sub_affined, sub_warped, label_inverse_affined, label_inverse_warped, sub2ref_affine, sub2ref_mapping

def loadLabels(f_label_vol, f_label_map):

    # load the image
    label_vol = nib.load(f_label_vol)

    # get the unique labels
    labels_in_vol = np.unique(label_vol.get_fdata())[1:].astype(int)  # skip the background label

    print(f" -- Found {len(labels_in_vol)} labels in the volume: {f_label_vol}")

    # read label map file
    print(f" -- Loading label map file: {f_label_map}")
    label_map_df = pd.read_csv(f_label_map, sep="\t")
    
    # check map file columns
    if "label" not in label_map_df.columns:
        raise ValueError("Text labels file does not contain 'label' column.")   
    if "name" not in label_map_df.columns:
        raise ValueError("Text labels file does not contain 'name' column.")
    
    
    labels_in_map = label_map_df["label"].to_list()
    
    # print the text labels
    print(f" -- Found {len(labels_in_map)} labels in the text file: {f_label_map}")
    
    # check if all the labels in the volume are in the text file
    if set(labels_in_vol).issubset(set(labels_in_map)):
        print(" -- All volume labels are present in the text label file.")
    else:
        print(" -- Some volume labels are not present in the text label file.") 
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
        print(f" -- Error loading nifti file: {nii_path}")
        print(f" -- {e}")
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

dataset = args.dataset ## dataset = "/home/nikhil/projects/Parkinsons/nimhans/data/ylo/"
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
print(f"Loading configuration file: {config_file}")
with open(config_file, 'r') as cf:
    config = json.load(cf)

# preproc config
preproc_pipeline_name = config.get("preproc_pipeline").get("name")
preproc_pipeline_version = config.get("preproc_pipeline").get("version")
preproc_pipeline_software = config.get("preproc_pipeline").get("software")

print(f"Using preproc pipeline: {preproc_pipeline_name} - version: {preproc_pipeline_version}")

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

if save_test_reg_images:    
    test_reg_images_dir = "./test_reg_images"
    Path(test_reg_images_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving registration QC images for testing at {test_reg_images_dir} for sanity checks.")

print("Running extraction of estimated features.")

# load label volume
print("Loading and verifying label volume.")

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
    reg_method = "affine" #  "affine" or "affine+syn"
else:
    reg_method = "affine+syn"

warp = not affine_only

print(f"Using registration method: {reg_method} --> warp={warp}")

# build input / output paths
datadir = pipeline_output_dir
outsdir = pipeline_idps_dir

# for every subject
print(f" -- Extracting IDP data from: {preproc_pipeline_name}-{preproc_pipeline_version}")

print(f" --  -- Processing: {bids_participant}")

# create output file name
outfile = Path(outsdir, f"{bids_participant}_{session}_{preproc_pipeline_name}-{preproc_pipeline_version}_{parc}_{reg_method}_idp.tsv")

# load the files to extract
print(f" --  -- Extracting data from: {bids_participant}")
dpdir = Path(datadir, bids_participant, f"{session}", "dipy")

mov = nib.load(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-fa_map.nii.gz"))

if save_test_reg_images:
    nib.save(mov, f'{test_reg_images_dir}/sub_orig.nii.gz')
    nib.save(ref, f'{test_reg_images_dir}/ref_orig.nii.gz')
    nib.save(labs, f'{test_reg_images_dir}/label_orig.nii.gz')

# linearly align dipy DTI FA (subject space) to the reference FA (JHU FA)
    
print(" --  --  -- Starting alignment and resampling.")  
# path to subject affine file
sub_aff_dir = f"{pipeline_output_dir}/{bids_participant}/{session}/affine/"
subj_aff_stem = f"{bids_participant}_{session}_{atlas_name}_sub2ref.txt"
subj_aff_path = Path(sub_aff_dir, subj_aff_stem)

# if the affine file exists, load it
sub2ref_affine = None
if subj_aff_path.exists():
    print(f" --  --  -- Using existing affine: {subj_aff_stem}")
    sub2ref_affine = np.loadtxt(subj_aff_path)

# path to subject warp file
sub_warp_dir = f"{pipeline_output_dir}/{bids_participant}/{session}/warp/"
subj_warp_stem = f"{bids_participant}_{session}_{atlas_name}_sub2ref_mapping.nii.gz"
subj_warp_path = Path(sub_warp_dir, subj_warp_stem)

# load existing warp if exists
sub2ref_mapping = None
if warp & (subj_warp_path.exists()):
    print(f"Using existing warp from: {subj_warp_path}")        
    sub2ref_mapping = read_mapping(str(subj_warp_path), mov, ref)

# # align and resample the moving image to the reference
sub_affined, sub_warped, label_inverse_affined, label_inverse_warped, sub2ref_affine, sub2ref_mapping = align_and_resample(
    mov, 
    ref, 
    labs, 
    affine_config=affine_registration_params,
    syn_config=None, 
    sub2ref_affine=sub2ref_affine, 
    sub2ref_mapping=sub2ref_mapping, 
    warp=warp
    )

print(" --  --  -- Alignment and resampling complete.")
# save the affine to disk
if not subj_aff_path.exists():
    Path(sub_aff_dir).mkdir(parents=True, exist_ok=True)
    print(f" --  --  -- Saving affine to: {subj_aff_path}")
    np.savetxt(subj_aff_path, sub2ref_affine)
    
# save warps
if warp & (not subj_warp_path.exists()):
    Path(sub_warp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving warp to: {subj_warp_path}")
    write_mapping(sub2ref_mapping, subj_warp_path)

# save the transformed images
if save_test_reg_images:
    nib.save(sub_affined, f'{test_reg_images_dir}/sub_affined.nii.gz')
    nib.save(label_inverse_affined, f'{test_reg_images_dir}/label_inverse_affined.nii.gz')
    if warp:
        nib.save(sub_warped, f'{test_reg_images_dir}/sub_warped.nii.gz')
        nib.save(label_inverse_warped, f'{test_reg_images_dir}/label_inverse_warped.nii.gz')
    
# get resampled labels
if warp:
    tldat = label_inverse_warped.get_fdata().astype(int)
else:
    tldat = label_inverse_affined.get_fdata().astype(int)

# load and prep data for extraction
# diff_params = ["fa","md","ad","nrmse","residual"]
diff_params = diffusion_metrics.keys()
print(f" --  --  -- Extracting diffusion metrics: {diff_params}")

# load data files
print(" --  --  -- Loading data files for extraction.")   
summary_df = pd.DataFrame() 
for diff_p in diff_params:
    diff_param_map, _ = load_nii(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-{diff_p}_map.nii.gz"), loader="dipy")
    diff_FW_param_map, _ = load_nii(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-{diff_p}_map.nii.gz"), loader="dipy")

    # for every roi label, get the mean value w/in the labels
    dti_roi_df = pd.DataFrame(columns=["roi_idx","model","param","mean_value","std_value","voxel_count"])
    fwdti_roi_df = pd.DataFrame(columns=["roi_idx","model","param","mean_value","std_value","voxel_count"])
    for idx, roi in enumerate(labels):
        # Default
        dti_val_mean, dti_val_std, dti_val_count = getROIStats(diff_param_map, tldat, roi)
        dti_roi_df.loc[idx] = [roi, "dti", diff_p, dti_val_mean, dti_val_std, dti_val_count]
        # FW
        FW_val_mean, FW_val_std, FW_val_count = getROIStats(diff_FW_param_map, tldat, roi)
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
print(f" --  -- Saved {bids_participant}_{session} IDPs to {outfile}.")

print("Done.")
