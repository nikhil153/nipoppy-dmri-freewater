import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import nibabel as nib
from nibabel.processing import conform

from dipy.align import affine_registration


def loadLabels(invol, tlabels):

    # load the image
    labs = nib.load(invol)

    # get the unique labels
    labels = np.unique(labs.get_fdata())[1:]

    # sanity check of text labels vs volume labels
    if all(tlabels["label"].isin(labels)):
        print(" -- Label names match the volume labels.")
    else:
        print(" -- Text label names do not match the volume label names.")
        raise ValueError("Text labels do not match the volume labels.")

    # pull the names from the text file
    roi_labels = tlabels["name"].to_list()

    # sort roi_labels by label number
    roi_labels = [x for _, x in sorted(zip(tlabels["label"], roi_labels))]

    return(labs, labels, roi_labels)


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


parser = argparse.ArgumentParser(description="load data by subject and create dataframes of stats to merge and plot.")
parser.add_argument("-d", "--derivatives", action='append', help='Derivatives directory(s) to crawl for IDP features', required=True)
# parser.add_argument("-p", "--subject", help="subject ID(s) to selectively extract", action='append')
parser.add_argument("-s", "--session", help="participant session to extract")
parser.add_argument("-r", "--label_ref", help="label volume to reference during alignment estimation")
parser.add_argument("-v", "--label_vol", help="label volume in reference space to extract ROI data")
parser.add_argument("-l", "--label_lab", help="label volume labels that match volume data")
parser.add_argument("-a", "--affine", help="file stem of affine to look for to shortcut alignment", default=None)
parser.add_argument("-f", "--force", nargs="?", const=1, type=bool, default=False, help="re-export all found subjects")
args = parser.parse_args()

pipelines = args.derivatives
# sids = args.subject
sess = args.session
label_ref = args.label_ref
label_vol = args.label_vol
label_lab = args.label_lab
label_aff = args.affine
redo = args.force

# pipelines = ["/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/derivatives/dmri_freewater_qsiprep", "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/derivatives/dmri_freewater_tractoflow"]
# subj = "sub-40533"
# sess = "BL"
# label_ref = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-FA.nii.gz"
# label_vol = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-tracts.nii.gz"
# label_lab = "/home/bcmcpher/Projects/nipoppy/qpn-subset-0.4.0/code/labels/JHU-tracts.tsv"
# label_aff = "JHU"

print("Running extraction of estimated features.")

#
# load label volume
#

print("Loading and verifying label volume.")

# load the label names to verify
tlabels = pd.read_csv(label_lab, sep="\t")

# check and load the label image
labs, labels, roi_labels = loadLabels(label_vol, tlabels)
parc = Path(label_vol).name.replace(".nii.gz", "")
nlabs = len(roi_labels)

# load the reference volume for coregistration
ref = nib.load(label_ref)

# the path to the shared affine transform
saffdir = Path(Path(label_vol).parent.absolute(), "affine")

#
# figure out pipeline paths to crawl and write to
#

# for each input pipeline
for pipe in pipelines:

    # get top level label for IDPs
    # pname = os.path.basename(pipe).replace("dmri_freewater_", "")
    # pvers = os.listdir(Path(pipe))[0]
    # print(f"Pipeline - Version: {pname}-{pvers}")
    pname = os.path.basename(pipe)
    pvers = "2.0.0"
    print(f"Pipeline - Version: {pname}-{pvers}")

    # build input / output paths
    datadir = Path(pipe, pvers, "output")
    outsdir = Path(pipe, pvers, "idps")

    # generator of subject folders
    subjs = os.listdir(Path(pipe, pvers, "output"))
    print(f" -- N subjects found: {len(list(subjs))}")

    # for every subject
    print(f" -- Extracting IDP data from: {pname}-{pvers}")
    for subj in subjs:

        # create sub ID w/o prefix
        sub = subj.replace("sub-", "")

        # create output file name
        outfile = Path(outsdir, f"sub-{sub}_ses-{sess}_{pname}-{pvers}_{parc}_idps.tsv")

        # if the file exists and no redo flag, skip iteration
        if outfile.exists() & (not redo):
            print(f" --  -- Data already extracted for: {subj}. Skipping.")
            continue

        # load the files to extract
        print(f" --  -- Extracting data from: {subj}")
        dpdir = Path(datadir, subj, f"ses-{sess}", "dipy")
        spdir = Path(datadir, subj, f"ses-{sess}", "scilpy")

        # loading bvals to check the kind of data
        try:
            bval = np.loadtxt(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_desc-fwcorr_dwi.bval"))
            if len(np.unique(bval)) < 3:
                print(" --  --  -- Single-shell data loaded")
                tshell = "single-shell"
            else:
                print(" --  --  -- Multi-shell data loaded")
                tshell = "multi-shell"
        except:
            print(" --  --  -- Bval not found?")
            tshell = "missing"

        #
        # linearly align dipy DTI FA (dpdtfa) to input ref
        #

        # path to subject affine file
        subj_aff_stem = f"sub-{sub}_ses-{sess}_{pname}-{pvers}_{label_aff}_sub2mni.txt"
        subj_aff = Path(saffdir, subj_aff_stem)

        # if the affine file exists, load it
        if subj_aff.exists():
            print(f" --  --  -- Using existing affine: {subj_aff_stem}")
            sub2mni = np.loadtxt(subj_aff)
            mni2sub = np.linalg.inv(sub2mni)
            mov = nib.load(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"))

        # otherwise compute alignment of FA to template
        else:
            print(" --  --  -- Affine alignment not found. Computing.")

            try:
                mov = nib.load(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"))
            except:
                print(" -- -- Failed to load DT FA image for alignment. Nothing can be done.")
                continue
            
            _, sub2mni = affine_registration(
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

        # save the affine to disk
        if not subj_aff.exists():
            saffdir.mkdir(parents=True, exist_ok=True)
            print(f" --  --  -- Saving affine to: {subj_aff}")
            np.savetxt(subj_aff, sub2mni)

        # get the invervse affine
        mni2sub = np.linalg.inv(sub2mni)

        # apply inverse of computed xfrom move ref labs to data space
        tlabs = conform(
            labs,
            out_shape=mov.shape,
            voxel_size=mov.header.get_zooms(),
            order=0,  # nearest neighbor
        )

        # get resampled labels
        tldat = tlabs.get_fdata()

        #
        # load and prep data for extraction
        #

        # load data files
        print(" --  --  -- Loading data files for extraction.")
        dpdtfa = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-fa_map.nii.gz"), tlabs)
        dpdtmd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-md_map.nii.gz"), tlabs)
        dpdtrd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-rd_map.nii.gz"), tlabs)
        dpdtad = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-ad_map.nii.gz"), tlabs)
        dpdtse = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-nrmse_map.nii.gz"), tlabs)
        dpdtrs = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-dti_param-residual_map.nii.gz"), tlabs)

        dpfwfa = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-fa_map.nii.gz"), tlabs)
        dpfwmd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-md_map.nii.gz"), tlabs)
        dpfwrd = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-rd_map.nii.gz"), tlabs)
        dpfwad = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-ad_map.nii.gz"), tlabs)
        dpfwfw = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-freewater_map.nii.gz"), tlabs)
        dpfwse = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-nrmse_map.nii.gz"), tlabs)
        dpfwrs = tryLoad(Path(dpdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-residual_map.nii.gz"), tlabs)

        spfwfa = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-fa_map.nii.gz"), tlabs)
        spfwmd = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-md_map.nii.gz"), tlabs)
        spfwrd = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-rd_map.nii.gz"), tlabs)
        spfwad = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-ad_map.nii.gz"), tlabs)
        spfwfw = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-freewater_map.nii.gz"), tlabs)
        spfwse = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-nrmse_map.nii.gz"), tlabs)
        spfwrs = tryLoad(Path(spdir, f"sub-{sub}_ses-{sess}_model-fwdti_param-residual_map.nii.gz"), tlabs)

        # preallocate output lists
        dpdtfa_mean = []
        dpdtmd_mean = [] 
        dpdtrd_mean = []
        dpdtad_mean = []
        dpdtfw_mean = np.zeros(nlabs)
        dpdtse_mean = []
        dpdtrs_mean = []

        dpfwfa_mean = []
        dpfwmd_mean = []
        dpfwrd_mean = []
        dpfwad_mean = []
        dpfwfw_mean = []
        dpfwse_mean = []
        dpfwrs_mean = []

        spfwfa_mean = []
        spfwmd_mean = []
        spfwrd_mean = []
        spfwad_mean = []
        spfwfw_mean = []
        spfwse_mean = []
        spfwrs_mean = []

        # for every roi label, get the mean value w/in the labels
        for idx, roi in enumerate(labels):
            dpdtfa_mean.append(nanAvg(dpdtfa[tldat == roi]))
            dpdtmd_mean.append(nanAvg(dpdtmd[tldat == roi]))
            dpdtrd_mean.append(nanAvg(dpdtrd[tldat == roi]))
            dpdtad_mean.append(nanAvg(dpdtad[tldat == roi]))
            dpdtse_mean.append(nanAvg(dpdtse[tldat == roi]))
            dpdtrs_mean.append(nanAvg(dpdtrs[tldat == roi]))

            dpfwfa_mean.append(nanAvg(dpfwfa[tldat == roi]))
            dpfwmd_mean.append(nanAvg(dpfwmd[tldat == roi]))
            dpfwrd_mean.append(nanAvg(dpfwrd[tldat == roi]))
            dpfwad_mean.append(nanAvg(dpfwad[tldat == roi]))
            dpfwfw_mean.append(nanAvg(dpfwfw[tldat == roi]))
            dpfwse_mean.append(nanAvg(dpfwse[tldat == roi]))
            dpfwrs_mean.append(nanAvg(dpfwrs[tldat == roi]))

            spfwfa_mean.append(nanAvg(spfwfa[tldat == roi]))
            spfwmd_mean.append(nanAvg(spfwmd[tldat == roi]))
            spfwrd_mean.append(nanAvg(spfwrd[tldat == roi]))
            spfwad_mean.append(nanAvg(spfwad[tldat == roi]))
            spfwfw_mean.append(nanAvg(spfwfw[tldat == roi]))
            spfwse_mean.append(nanAvg(spfwse[tldat == roi]))
            spfwrs_mean.append(nanAvg(spfwrs[tldat == roi]))

        #
        # create output dataframes
        #

        # merge regular dipy tensor
        dpdt_data = pd.DataFrame([roi_labels,
                                  dpdtfa_mean,
                                  dpdtmd_mean,
                                  dpdtrd_mean,
                                  dpdtad_mean,
                                  dpdtfw_mean,
                                  dpdtse_mean,
                                  dpdtrs_mean])
        dpdt_data = dpdt_data.T

        dpdt_data["subj"] = sub
        dpdt_data["pipe"] = f"{pname}-{pvers}"
        dpdt_data["soft"] = "dipy"
        dpdt_data["shell"] = tshell
        dpdt_data["model"] = "dti"

        # label and reorder columns
        dpdt_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        dpdt_data = dpdt_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge fw dipy tensor
        dpfw_data = pd.DataFrame([roi_labels,
                                  dpfwfa_mean,
                                  dpfwmd_mean,
                                  dpfwrd_mean,
                                  dpfwad_mean,
                                  dpfwfw_mean,
                                  dpfwse_mean,
                                  dpfwrs_mean])
        dpfw_data = dpfw_data.T

        dpfw_data["subj"] = sub
        dpfw_data["pipe"] = f"{pname}-{pvers}"
        dpfw_data["soft"] = "dipy"
        dpfw_data["shell"] = tshell
        dpfw_data["model"] = "fwdti"

        # label and reorder columns
        dpfw_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        dpfw_data = dpfw_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge scilpy fw tensor
        spfw_data = pd.DataFrame([roi_labels,
                                  spfwfa_mean,
                                  spfwmd_mean,
                                  spfwrd_mean,
                                  spfwad_mean,
                                  spfwfw_mean,
                                  spfwse_mean,
                                  spfwrs_mean])
        spfw_data = spfw_data.T

        spfw_data["subj"] = sub
        spfw_data["pipe"] = f"{pname}-{pvers}"
        spfw_data["soft"] = "scilpy"
        spfw_data["shell"] = tshell
        spfw_data["model"] = "fwdti"

        # label and reorder columns
        spfw_data.columns = ["roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual", "subj", "pipeline", "software", "shell", "model"]
        spfw_data = spfw_data[["subj", "pipeline", "software", "shell", "model", "roi_labels", "fa", "md", "rd", "ad", "fw", "nrmse", "residual"]]

        # merge the dataframes
        sdata = pd.concat([dpdt_data, dpfw_data, spfw_data])

        # write the dataframe to disk
        sdata.to_csv(outfile, index=False, header=True, sep="\t")
        print(f" --  -- Saved sub-{sub}_ses-{sess} IDPs to disk.")

print("Done.")
