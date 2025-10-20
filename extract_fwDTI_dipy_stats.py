import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import nibabel as nib
from nibabel.processing import conform

from dipy.align import affine_registration

from os.path import join as pjoin

import numpy as np

from dipy.align import affine_registration, syn_registration, write_mapping, read_mapping
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.imaffine import (
    AffineMap,
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.viz import regtools

def warp_and_resample(mov, ref, label=None, sub2ref_mapping=None):
    """
    SyN registration and creation of DiffeomorphicMap.

    Parameters
    ----------
    mov : nibabel.Nifti1Image
        Moving image.
    ref : nibabel.Nifti1Image
        Reference image.

    Returns
    -------
    
    """

    if sub2ref_mapping is not None:
        print(" --  --  -- Using provided warp for registration.")

    else:
        _, sub2ref_mapping = syn_registration(mov, ref)


    # Transform the subject image data using the diffeomorphic map
    sub_transformed_data = sub2ref_mapping.transform(mov.get_fdata())
    sub_transformed = nib.Nifti1Image(sub_transformed_data, ref.affine)

    # Transform the ref image data using the diffeomorphic map
    ref_transformed_data = sub2ref_mapping.transform_inverse(ref.get_fdata())
    ref_transformed = nib.Nifti1Image(ref_transformed_data, mov.affine)

    if label is not None:
        # Transform the ref label data using the diffeomorphic map
        label_transformed_data = sub2ref_mapping.transform_inverse(label.get_fdata(), interpolation='nearest')
        label_transformed = nib.Nifti1Image(label_transformed_data, mov.affine)


    return sub_transformed, ref_transformed, label_transformed, sub2ref_mapping


def align_and_resample(mov, ref, label=None, sub2ref=None):
    """
    Affine registration of moving image to static image and resampling of moving image to static image space.

    Parameters
    ----------
    mov : nibabel.Nifti1Image
        The moving image to be registered and resampled.
    ref : nibabel.Nifti1Image
        The static reference image.

    Returns
    -------
    sub_transformed : nibabel.Nifti1Image
        The registered and resampled moving image.
    ref_transformed : nibabel.Nifti1Image
        The registered and resampled static image.
    sub2ref : np.ndarray
        The affine transformation matrix from moving to static image space.
    """

    # Affine registration

    if sub2ref is not None:
        print(" --  --  -- Using provided affine for registration.")

    else:
        print(" --  --  -- Computing affine registration.")
        _, sub2ref = affine_registration(
            mov,
            ref,
            moving_affine=mov.affine,
            static_affine=ref.affine,
            nbins=32,
            metric="MI",
            pipeline=["center_of_mass", "translation", "rigid", "affine"],
            level_iters=[10000, 1000, 100],
            sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1],
        )

    # Transform the moving image
    affmap = AffineMap(
        sub2ref,
        ref.shape,
        ref.affine,
        mov.shape,
        mov.affine,
    )

    # apply the transformation to the moving image data
    sub_transformed_data = affmap.transform(mov.get_fdata())
    sub_transformed = nib.Nifti1Image(sub_transformed_data, ref.affine)

    # apply inverse transformation to the reference image data
    ref_transformed_data = affmap.transform_inverse(ref.get_fdata())
    ref_transformed = nib.Nifti1Image(ref_transformed_data, mov.affine)

    if label is not None:
        # resample the label image to the moving image space
        label_transformed_data = affmap.transform_inverse(label.get_fdata(), interpolation='nearest')
        label_transformed = nib.Nifti1Image(label_transformed_data, mov.affine) 
    else:
        label_transformed = None

    return sub_transformed, ref_transformed, label_transformed, sub2ref

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


def tryLoad(invol, labs):
    try:

        dat = nib.load(invol)
        vol = dat.get_fdata()
        out = vol
        # print(f" --  --  -- Successfully loaded: {invol}")

    except:

        out = np.empty(labs.shape)
        out[:] = np.nan
        # print(f" --  --  -- Failed to load: {invol}")

    return out


def nanAvg(inp):

    if inp.size == 0:
        out = np.nan

    else:
        try:
            out = np.nanmean(inp)
        except:
            out = np.nan

    return(out)


# parser = argparse.ArgumentParser(description="load data by subject and create dataframes of stats to merge and plot.")
# parser.add_argument("-d", "--derivative_dir", help='Derivatives directory contained processed dMRI data', required=True)
# parser.add_argument("-p", "--participant_id", help="participant ID(s)", nargs="+", default=None)
# parser.add_argument("-s", "--session", help="participant session", default="01")
# parser.add_argument("-i", "--ref_img", help="reference img for alignment estimation")
# parser.add_argument("-l", "--ref_label", help="reference label volume to extract ROI data")
# parser.add_argument("-m", "--label_map", help="tract labels that match volume data")
# parser.add_argument("-a", "--affine", help="file stem of affine to look for to shortcut alignment", default=None)
# parser.add_argument("-f", "--force", nargs="?", const=1, type=bool, default=False, help="re-export all found subjects")
# args = parser.parse_args()

# pipelines = args.derivative_dir
# participant_id = args.participant_id
# session_id = args.session

# ref_img = args.ref_img
# ref_label = args.ref_label
# label_map = args.label_map

# label_aff = args.affine
# redo = args.force

# dataset directory
dataset = "/home/nikhil/projects/Parkinsons/nimhans/data/ylo/"
pipeline_output_dir = f"{dataset}/derivatives/dmri-freewater/2.0.0/output/"
pipeline_idps_dir = f"{dataset}/derivatives/dmri-freewater/2.0.0/idp/"

bids_participant_list = ["sub-YLOPD31"] #,"sub-YLOPD321"]
session_id = "01"
save_test_reg_images = False
test_reg_images_dir = "./test_reg_images"

# check is session id has sub- prefix
if session_id.startswith("ses-"):
    session_id = session_id.replace("ses-", "")
    
session = f"ses-{session_id}"

# labels and reference volumes
labels_dir = "/home/nikhil/projects/neuroinformatics_tools/nipoppy-dmri-freewater/atlases/"

ref_img = f"{labels_dir}/JHU-ICBM-FA-1mm.nii.gz"
ref_label = f"{labels_dir}/JHU-ICBM-labels-1mm.nii.gz"
label_map = f"{labels_dir}/JHU-ICBM-tract-label_desc.tsv"
label_aff = "JHU"

print("Running extraction of estimated features.")

# load label volume
print("Loading and verifying label volume.")

# check and load the label image
labs, labels, roi_labels = loadLabels(ref_label, label_map)
parc = Path(ref_label).name.replace(".nii.gz", "")
nlabs = len(roi_labels)

# load the reference volume for coregistration
ref = nib.load(ref_img)

# the path to the shared affine transform
saffdir = Path(Path(ref_label).parent.absolute(), "affine")

# get top level label for IDPs
pname = "qsiprep-fw"
pvers = "2.0.0"
print(f"Pipeline - Version: {pname}-{pvers}")

tshell = "multi-shell" # ylo is all multi-shell
reg_method = "syn" # "syn" or "affine" 

# build input / output paths
datadir = pipeline_output_dir
outsdir = pipeline_idps_dir

# for every subject
print(f" -- Extracting IDP data from: {pname}-{pvers}")
for bids_participant in bids_participant_list:
    
    # check the sub prefix
    if not bids_participant.startswith("sub-"):
        bids_participant = f"sub-{bids_participant}"

    print(f" --  -- Processing: {bids_participant}")

    # create output file name
    outfile = Path(outsdir, f"{bids_participant}_{session}_{pname}-{pvers}_{parc}_{reg_method}_idp.tsv")

    # load the files to extract
    print(f" --  -- Extracting data from: {bids_participant}")
    dpdir = Path(datadir, bids_participant, f"{session}", "dipy")
    spdir = Path(datadir, bids_participant, f"{session}", "scilpy")


    if save_test_reg_images:
        nib.save(mov, f'{test_reg_images_dir}/sub_orig.nii.gz')
        nib.save(ref, f'{test_reg_images_dir}/ref_orig.nii.gz')
        nib.save(labs, f'{test_reg_images_dir}/label_orig.nii.gz')
    

    if reg_method == "affine":
        # linearly align dipy DTI FA (dpdtfa) to input ref

        # path to subject affine file
        subj_aff_stem = f"{bids_participant}_{session}_{label_aff}_sub2ref.txt"
        subj_aff_path = Path(saffdir, subj_aff_stem)

        # if the affine file exists, load it
        sub2ref = None
        if subj_aff_path.exists():
            print(f" --  --  -- Using existing affine: {subj_aff_stem}")
            sub2ref = np.loadtxt(subj_aff_path)
        
        mov = nib.load(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-fa_map.nii.gz"))

        # align and resample the moving image to the reference
        sub_transformed, ref_transformed, label_transformed, sub2ref = align_and_resample(mov, ref, labs, sub2ref)

        # save the affine to disk
        if not subj_aff_path.exists():
            saffdir.mkdir(parents=True, exist_ok=True)
            print(f" --  --  -- Saving affine to: {subj_aff_path}")
            np.savetxt(subj_aff_path, sub2ref)

        if save_test_reg_images:
            # save the transformed images
            nib.save(sub_transformed, f'{test_reg_images_dir}/sub_affine_transformed.nii.gz')
            nib.save(ref_transformed, f'{test_reg_images_dir}/ref_affine_transformed.nii.gz')
            nib.save(label_transformed, f'{test_reg_images_dir}/label_affine_transformed.nii.gz')

    elif reg_method == "syn":

        # path to subject warp file
        subj_warp_stem = f"{bids_participant}_{session}_{label_aff}_sub2ref_warp.nii.gz"
        subj_warp_path = Path(saffdir, subj_warp_stem)

        # load existing warp if exists
        sub2ref = read_mapping(subj_warp_path, ref, mov)

        # syn registration
        print("Syn registration started")
        sub_transformed, ref_transformed, label_transformed, sub2ref = warp_and_resample(mov, ref, labs)
        print("Syn registration complete")

        # save warps
        write_mapping(sub2ref, warp_path)

        if save_test_reg_images:
            nib.save(sub_transformed, f'{test_reg_images_dir}/sub_syn_transformed.nii.gz')
            nib.save(ref_transformed, f'{test_reg_images_dir}/ref_syn_transformed.nii.gz')
            nib.save(label_transformed, f'{test_reg_images_dir}/label_syn_transformed.nii.gz')

        

    

    # get resampled labels
    tldat = label_transformed.get_fdata()

    # load and prep data for extraction

    # load data files
    print(" --  --  -- Loading data files for extraction.")
    dpdtfa = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-fa_map.nii.gz"), label_transformed)
    dpdtse = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-nrmse_map.nii.gz"), label_transformed)
    dpdtrs = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-dti_param-residual_map.nii.gz"), label_transformed)
    dpfwfa = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-fa_map.nii.gz"), label_transformed)
    dpfwfw = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-freewater_map.nii.gz"), label_transformed)
    dpfwse = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-nrmse_map.nii.gz"), label_transformed)
    dpfwrs = tryLoad(Path(dpdir, f"{bids_participant}_{session}_model-fwdti_param-residual_map.nii.gz"), label_transformed)

    # preallocate output lists
    dpdtfa_mean = []
    dpdtfw_mean = np.zeros(nlabs)
    dpdtse_mean = []
    dpdtrs_mean = []

    dpfwfa_mean = []
    dpfwfw_mean = []
    dpfwse_mean = []
    dpfwrs_mean = []

    # for every roi label, get the mean value w/in the labels
    for idx, roi in enumerate(labels):
        dpdtfa_mean.append(nanAvg(dpdtfa[tldat == roi]))
        dpdtse_mean.append(nanAvg(dpdtse[tldat == roi]))
        dpdtrs_mean.append(nanAvg(dpdtrs[tldat == roi]))

        dpfwfa_mean.append(nanAvg(dpfwfa[tldat == roi]))
        dpfwfw_mean.append(nanAvg(dpfwfw[tldat == roi]))
        dpfwse_mean.append(nanAvg(dpfwse[tldat == roi]))
        dpfwrs_mean.append(nanAvg(dpfwrs[tldat == roi]))


    # merge regular dipy tensor
    dpdt_data = pd.DataFrame([roi_labels,
                                dpdtfa_mean,
                                dpdtfw_mean,
                                dpdtse_mean,
                                dpdtrs_mean])
    dpdt_data = dpdt_data.T

    dpdt_data["participant_id"] = bids_participant
    dpdt_data["pipeline"] = f"{pname}-{pvers}"
    dpdt_data["software"] = "dipy"
    dpdt_data["shell"] = tshell
    dpdt_data["model"] = "dti"
    dpdt_data["registration"] = reg_method


    # label and reorder columns
    dpdt_data.columns = ["roi_labels", "fa", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
    dpdt_data = dpdt_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "fw", "nrmse", "residual"]]

    # merge fw dipy tensor
    dpfw_data = pd.DataFrame([roi_labels,
                                dpfwfa_mean,
                                dpfwfw_mean,
                                dpfwse_mean,
                                dpfwrs_mean])
    dpfw_data = dpfw_data.T

    dpfw_data["subj"] = bids_participant
    dpfw_data["pipe"] = f"{pname}-{pvers}"
    dpfw_data["soft"] = "dipy"
    dpfw_data["shell"] = tshell
    dpfw_data["model"] = "fwdti"

    # label and reorder columns
    dpfw_data.columns = ["roi_labels", "fa", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
    dpfw_data = dpfw_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "fw", "nrmse", "residual"]]

    # merge the dataframes
    idp_data = pd.concat([dpdt_data, dpfw_data])

    # write the dataframe to disk
    idp_data.to_csv(outfile, index=False, header=True, sep="\t")
    print(f" --  -- Saved {bids_participant}_{session} IDPs to disk.")

print("Done.")
