import os, re, gc
from glob import glob
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

from nilearn import image as nimg
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------
# Processing Function
# ---------------------------------------------------------
def _label_from_bids(basename: str) -> str:
    m = re.search(r'_task-(trauma_run-\d+)_', basename)
    if not m:
        raise ValueError(f"Expected run-* in filename: {basename}")
    return m.group(1)

def process_one(f, mask_path, atlas_name):
    """Return (sub, ok_bool, msg)."""
    import warnings
    import traceback
    warnings.filterwarnings('ignore')
    
    try:
        base  = os.path.basename(f)
        sub   = base.split('_')[0]              
        label = _label_from_bids(base)          

        out_csv = os.path.join(os.path.dirname(f), f"{sub}_{label}_Average_ROI_{atlas_name}.csv")
        
        if os.path.exists(out_csv):
            return sub, True, f'SKIP (exists: {os.path.basename(out_csv)})'

        # Initialize masker inside the worker to prevent OOM pickling bloat
        if atlas_name == 'Neurosynth':
            masker = NiftiLabelsMasker(labels_img=mask_path, standardize=None, detrend=False)
        else:
            masker = NiftiMapsMasker(maps_img=mask_path, standardize=None, detrend=False)

        # Extract Time Series
        time_series = masker.fit_transform(f)

        # Write to CSV
        pd.DataFrame(time_series).to_csv(out_csv, index=False)
        
        # Aggressive memory cleanup
        del time_series, masker
        gc.collect()

        return sub, True, f'WROTE {os.path.basename(out_csv)}'
        
    except Exception as e:
        tb = traceback.format_exc()
        return base.split('_')[0], False, f'{e.__class__.__name__}: {e}\n{tb}'

# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == '__main__':
    # Keep BLAS threads in check on HPC to prevent CPU throttling
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    atlas = 'Neurosynth' # Options: 'Neurosynth' or 'difumo'
    HPC = '/nfs/roberts/pi/pi_il77/Nachshon/'

    if atlas == 'Neurosynth':
        mask_path = f'{HPC}ROI/Atlas/NeurosynthParcellation/Neurosynth Parcellation_0.nii.gz'
    else:
        mask_path = f'{HPC}ROI/Atlas/difumo_2mm_maps.nii.gz'

    print(f"Loaded {atlas} atlas path: {mask_path}")

    output_dir = f'{HPC}/PSUB/preProc_o/'
    func_files = glob(os.path.join(output_dir, 'sub-*_ses-1_task-trauma_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_denoise_smooth6mm_bold.nii.gz'))
    print(f'Found {len(func_files)} files')

    # We map this to the SLURM CPUS_PER_TASK variable, defaulting to 4 if not found
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f'Using {n_jobs} workers')

    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(process_one)(f, mask_path, atlas) for f in func_files
    )

    # ---- report ----
    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    print(f'\nExtracted: {len(ok)}')
    for sub, _, msg in ok:
        print(f'[OK]   {sub} -> {msg}')

    if fail:
        print(f'\nFailed: {len(fail)}')
        for sub, _, msg in fail:
            print(f'[FAIL] {sub} -> {msg}')