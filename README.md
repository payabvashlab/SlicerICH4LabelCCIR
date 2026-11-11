# ICH 4-Label ORBIT-CCIR for 3D Slicer

This 3D Slicer scripted extension deploys the combined method2 + method1
pipeline as one user-facing workflow:

1. Use one NCCT volume loaded in Slicer, or recursively discover a folder of
   NIfTI CT volumes.
2. Run the trained method2 ORBIT-RSNA nnU-Net with labels '1=ICH',
   '2=Edema', '3=Ventricle', and '4=IVH'.
3. Find 26-connected predicted-IVH components.
4. Check only components smaller than 1.0 mL with the trained method1
   Component-Conditioned IVH Refiner (CCIR).
5. Remove low-confidence components, retain uncertain components, and
   optionally use the residual head around high-confidence components.
6. Save final segmentations and reports in the selected output folder. A
   single-volume result is also loaded into Slicer.

Labels 1, 2, and 3 are copied unchanged from the ORBIT-RSNA baseline. CCIR
modifies only IVH (label 4) and background. A hard runtime invariant stops
output saving if any protected label changes.

# Important model requirement

The extension contains inference code but not private trained weights. You need:

- the trained method2 ORBIT-RSNA results Dataset/configuration folder;
- 'ccir_best_resubstitution.pt' produced after training CCIR from method2
  predictions.

# Install in Slicer

### Development installation

1. Extract this archive.
2. Open 3D Slicer.
3. Drag and drop python code to 3D Slicer
4. Restart Slicer and open Segmentation > ICH 4-Label ORBIT-CCIR.
5. Expand Runtime, click Install/verify Python dependencies, then restart
   Slicer once more.

# Run

1. Choose Loaded volume and select one NCCT, or choose NIfTI folder
   (batch) and select an input folder.
2. Select an Output save folder. Single and batch outputs are written there.
3. Select the method2 ORBIT-RSNA Dataset or configuration folder.
4. Select orbit_rsna_ccir_all_train_all_validate/ccir_best_resubstitution.pt.
5. Enter the calibration probabilities directly in the GUI.
6. Keep Component volume gate = 1.00 mL, unless another gate was validated.
7. Select the preferred physical CUDA GPU (default 0). If CUDA is unavailable,
   the worker automatically runs both stages on CPU.
8. Click Run ORBIT-RSNA + CCIR.

## Threshold semantics

- 'component volume gate = 1.0 mL': run CCIR only when a predicted IVH
  component is smaller than 1 mL. It is not a probability threshold.
- 'remove probability': if 'P(true IVH)' is below this value, label 4 becomes
  background for that component.
- 'keep/refine probability': if probability is at least this value, preserve the
  component and optionally add residual IVH voxels locally.
- Between the two probability thresholds, abstain and preserve the method2
  ORBIT-RSNA component.
