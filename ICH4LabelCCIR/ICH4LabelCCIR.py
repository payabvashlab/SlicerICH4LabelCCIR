from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class ICH4LabelCCIR(ScriptedLoadableModule):
    """ORBIT-RSNA four-label segmentation plus CCIR IVH refinement."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "ICH 4-Label ORBIT-CCIR"
        self.parent.categories = ["Segmentation"]
        # Do not declare PyTorch as a scripted-module load dependency here.
        # Otherwise Slicer refuses to load this module when PyTorch is absent,
        # and the first-run installer UI below can never be reached. Packaged
        # extension installation is handled by EXTENSION_DEPENDS in CMake;
        # developer/source-tree installs are handled at runtime by this module.
        self.parent.dependencies = []
        self.parent.contributors = ["Anh Tuan Tran"]
        self.parent.helpText = (
            "Runs the trained method2 ORBIT-RSNA four-label model (ICH, edema, "
            "ventricle, IVH), then applies the method1 Component-Conditioned IVH "
            "Refiner. Labels 1-3 remain exactly unchanged."
        )
        self.parent.acknowledgementText = (
            "nnU-Net: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. "
            "Nature Methods. 2021;18:203-211."
        )


class ICH4LabelCCIRWidget(ScriptedLoadableModuleWidget):
    SETTINGS_PREFIX = "ICH4LabelCCIR"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic: Optional[ICH4LabelCCIRLogic] = None
        self._process = None
        self._temporaryDirectory = None

    def setup(self):
        super().setup()
        self.logic = ICH4LabelCCIRLogic()

        instructions = qt.QLabel(
            "Select a non-contrast head CT, the trained method2 ORBIT-RSNA Dataset "
            "folder, and the CCIR checkpoint trained from those method2 predictions. "
            "The volume gate is not a probability threshold."
        )
        instructions.wordWrap = True
        self.layout.addWidget(instructions)

        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Input and models"
        self.layout.addWidget(inputCollapsible)
        inputForm = qt.QFormLayout(inputCollapsible)

        self.inputModeComboBox = qt.QComboBox()
        self.inputModeComboBox.addItems(["Loaded volume", "NIfTI folder (batch)"])
        inputForm.addRow("Input mode:", self.inputModeComboBox)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = False
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        inputForm.addRow("NCCT volume:", self.inputSelector)
        self.inputVolumeLabel = inputForm.labelForField(self.inputSelector)

        self.inputDirectory = ctk.ctkPathLineEdit()
        self.inputDirectory.filters = ctk.ctkPathLineEdit.Dirs
        self.inputDirectory.settingKey = f"{self.SETTINGS_PREFIX}/inputDirectory"
        self.inputDirectory.toolTip = (
            "Batch mode: recursively segment every .nii or .nii.gz CT in this folder. "
            "Files ending in _seg, _label, _mask, or channels _0001 and above are ignored."
        )
        inputForm.addRow("Input NIfTI folder:", self.inputDirectory)
        self.inputDirectoryLabel = inputForm.labelForField(self.inputDirectory)

        self.outputDirectory = ctk.ctkPathLineEdit()
        self.outputDirectory.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDirectory.settingKey = f"{self.SETTINGS_PREFIX}/outputDirectory"
        self.outputDirectory.toolTip = (
            "Final *_seg.nii.gz files and JSON reports are saved here."
        )
        inputForm.addRow("Output save folder:", self.outputDirectory)

        self.overwriteCheckBox = qt.QCheckBox("Overwrite existing output segmentations")
        self.overwriteCheckBox.checked = False
        inputForm.addRow("", self.overwriteCheckBox)

        self.modelDirectory = ctk.ctkPathLineEdit()
        self.modelDirectory.filters = ctk.ctkPathLineEdit.Dirs
        self.modelDirectory.settingKey = f"{self.SETTINGS_PREFIX}/nnUNetModelDirectory"
        self.modelDirectory.toolTip = (
            "Dataset516_ICH_CT_ResEncL_MedNeXtLk3_4labels_full from method2. It "
            "must contain dataset.json and the ORBIT trainer configuration folder."
        )
        inputForm.addRow("Method2 ORBIT Dataset folder:", self.modelDirectory)

        self.ccirCheckpoint = ctk.ctkPathLineEdit()
        self.ccirCheckpoint.filters = ctk.ctkPathLineEdit.Files
        self.ccirCheckpoint.nameFilters = ["PyTorch checkpoint (*.pt *.pth)", "All files (*)"]
        self.ccirCheckpoint.settingKey = f"{self.SETTINGS_PREFIX}/ccirCheckpoint"
        self.ccirCheckpoint.toolTip = (
            "Select orbit_rsna_ccir_all_train_all_validate/"
            "ccir_best_resubstitution.pt. Do not select RSNA classifier_best.pt."
        )
        inputForm.addRow("CCIR checkpoint:", self.ccirCheckpoint)

        modelCollapsible = ctk.ctkCollapsibleButton()
        modelCollapsible.text = "Method2 ORBIT-RSNA inference settings"
        modelCollapsible.collapsed = True
        self.layout.addWidget(modelCollapsible)
        modelForm = qt.QFormLayout(modelCollapsible)

        self.configurationEdit = qt.QLineEdit("3d_fullres")
        self.trainerEdit = qt.QLineEdit("nnUNetTrainerORBITRSNA")
        self.plansEdit = qt.QLineEdit("nnUNetResEncUNetLPlans")
        self.foldsEdit = qt.QLineEdit("all")
        self.nnUNetCheckpointEdit = qt.QLineEdit("checkpoint_best.pth")
        modelForm.addRow("Configuration:", self.configurationEdit)
        modelForm.addRow("Trainer:", self.trainerEdit)
        modelForm.addRow("Plans:", self.plansEdit)
        modelForm.addRow("Fold(s):", self.foldsEdit)
        modelForm.addRow("nnU-Net checkpoint:", self.nnUNetCheckpointEdit)

        refinerCollapsible = ctk.ctkCollapsibleButton()
        refinerCollapsible.text = "IVH gate and calibration settings (GUI)"
        self.layout.addWidget(refinerCollapsible)
        refinerForm = qt.QFormLayout(refinerCollapsible)

        self.volumeGateSpinBox = qt.QDoubleSpinBox()
        self.volumeGateSpinBox.minimum = 0.01
        self.volumeGateSpinBox.maximum = 20.0
        self.volumeGateSpinBox.decimals = 2
        self.volumeGateSpinBox.singleStep = 0.1
        self.volumeGateSpinBox.value = 1.0
        self.volumeGateSpinBox.suffix = " mL"
        self.volumeGateSpinBox.toolTip = (
            "Only predicted IVH connected components smaller than this physical volume "
            "are checked by CCIR. Components at or above the gate are preserved."
        )
        refinerForm.addRow("Component volume gate:", self.volumeGateSpinBox)

        calibrationLabel = qt.QLabel(
            "Calibration probabilities are entered here and passed directly to "
            "the worker; no calibration JSON file is loaded."
        )
        calibrationLabel.wordWrap = True
        refinerForm.addRow(calibrationLabel)

        self.removeThresholdSpinBox = qt.QDoubleSpinBox()
        self.removeThresholdSpinBox.minimum = 0.0
        self.removeThresholdSpinBox.maximum = 1.0
        self.removeThresholdSpinBox.decimals = 4
        self.removeThresholdSpinBox.singleStep = 0.01
        self.removeThresholdSpinBox.value = 0.5
        self.removeThresholdSpinBox.toolTip = (
            "A checked component is removed when P(true IVH) is below this threshold. "
            "This GUI value is used directly; no calibration-report file is read."
        )
        refinerForm.addRow("Remove probability:", self.removeThresholdSpinBox)

        self.keepThresholdSpinBox = qt.QDoubleSpinBox()
        self.keepThresholdSpinBox.minimum = 0.0
        self.keepThresholdSpinBox.maximum = 1.0
        self.keepThresholdSpinBox.decimals = 4
        self.keepThresholdSpinBox.singleStep = 0.01
        self.keepThresholdSpinBox.value = 0.9
        self.keepThresholdSpinBox.toolTip = (
            "A checked component at or above this probability may use the residual "
            "segmentation head. Intermediate probabilities preserve the method2 result."
        )
        refinerForm.addRow("Keep/refine probability:", self.keepThresholdSpinBox)

        self.disableResidualCheckBox = qt.QCheckBox("Disable residual voxel addition")
        self.disableResidualCheckBox.checked = False
        refinerForm.addRow("", self.disableResidualCheckBox)

        runtimeCollapsible = ctk.ctkCollapsibleButton()
        runtimeCollapsible.text = "Runtime"
        runtimeCollapsible.collapsed = True
        self.layout.addWidget(runtimeCollapsible)
        runtimeForm = qt.QFormLayout(runtimeCollapsible)

        self.gpuSpinBox = qt.QSpinBox()
        self.gpuSpinBox.minimum = 0
        self.gpuSpinBox.maximum = 15
        self.gpuSpinBox.value = 0
        self.gpuSpinBox.toolTip = (
            "Preferred physical CUDA GPU index. If CUDA is unavailable, inference "
            "automatically falls back to CPU."
        )
        runtimeForm.addRow("Physical GPU:", self.gpuSpinBox)

        self.nppSpinBox = qt.QSpinBox()
        self.nppSpinBox.minimum = 1
        self.nppSpinBox.maximum = 16
        self.nppSpinBox.value = 1
        self.npsSpinBox = qt.QSpinBox()
        self.npsSpinBox.minimum = 1
        self.npsSpinBox.maximum = 16
        self.npsSpinBox.value = 1
        runtimeForm.addRow("Preprocessing workers:", self.nppSpinBox)
        runtimeForm.addRow("Export workers:", self.npsSpinBox)

        self.disableTtaCheckBox = qt.QCheckBox("Disable nnU-Net mirroring/TTA")
        self.disableTtaCheckBox.checked = False
        runtimeForm.addRow("", self.disableTtaCheckBox)

        self.installButton = qt.QPushButton("Install/verify Python dependencies")
        self.installButton.toolTip = (
            "Install nnU-Net, MONAI, nibabel, SciPy and pandas into Slicer's Python. "
            "Install the Slicer PyTorch extension first."
        )
        runtimeForm.addRow(self.installButton)

        actionRow = qt.QHBoxLayout()
        self.applyButton = qt.QPushButton("Run ORBIT-RSNA + CCIR")
        self.applyButton.enabled = False
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        actionRow.addWidget(self.applyButton)
        actionRow.addWidget(self.cancelButton)
        self.layout.addLayout(actionRow)

        self.statusLabel = qt.QLabel("Ready")
        self.layout.addWidget(self.statusLabel)
        self.logText = qt.QPlainTextEdit()
        self.logText.readOnly = True
        self.logText.minimumHeight = 220
        self.layout.addWidget(self.logText)
        self.layout.addStretch(1)

        self.inputModeComboBox.connect("currentIndexChanged(int)", self._onInputModeChanged)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._updateApplyState)
        self.inputDirectory.connect("currentPathChanged(QString)", self._updateApplyState)
        self.outputDirectory.connect("currentPathChanged(QString)", self._updateApplyState)
        self.modelDirectory.connect("currentPathChanged(QString)", self._updateApplyState)
        self.ccirCheckpoint.connect("currentPathChanged(QString)", self._updateApplyState)
        self.installButton.connect("clicked()", self._installDependencies)
        self.applyButton.connect("clicked()", self._run)
        self.cancelButton.connect("clicked()", self._cancel)
        self._onInputModeChanged()

    def cleanup(self):
        self._cancel()

    def _appendLog(self, message: str):
        if not message:
            return
        self.logText.appendPlainText(message.rstrip())
        # PythonQt's QTextCursor wrapper can report invalid negative positions
        # while QProcess output is appended rapidly. Scrolling through the
        # vertical scroll bar avoids manipulating a transient cursor.
        scrollBar = self.logText.verticalScrollBar()
        scrollBar.setValue(scrollBar.maximum)
        slicer.app.processEvents()

    def _onInputModeChanged(self, *_):
        batchMode = self.inputModeComboBox.currentIndex == 1
        self.inputSelector.setVisible(not batchMode)
        self.inputVolumeLabel.setVisible(not batchMode)
        self.inputDirectory.setVisible(batchMode)
        self.inputDirectoryLabel.setVisible(batchMode)
        self._updateApplyState()

    def _updateApplyState(self, *_):
        if not hasattr(self, "applyButton"):
            return
        batchMode = self.inputModeComboBox.currentIndex == 1
        inputDirectoryText = self.inputDirectory.currentPath.strip()
        outputDirectoryText = self.outputDirectory.currentPath.strip()
        modelDirectoryText = self.modelDirectory.currentPath.strip()
        checkpointText = self.ccirCheckpoint.currentPath.strip()
        validInput = (
            bool(inputDirectoryText) and Path(inputDirectoryText).is_dir()
            if batchMode
            else self.inputSelector.currentNode() is not None
        )
        valid = (
            validInput
            and bool(outputDirectoryText)
            and Path(outputDirectoryText).is_dir()
            and bool(modelDirectoryText)
            and Path(modelDirectoryText).is_dir()
            and bool(checkpointText)
            and Path(checkpointText).is_file()
            and self._process is None
        )
        self.applyButton.enabled = bool(valid)

    def _installDependencies(self):
        if self._process is not None:
            return
        try:
            missing = self.logic.missingDependencies()
            if not missing:
                self.statusLabel.text = "All Python dependencies are available"
                slicer.util.infoDisplay("All required Python packages are installed.")
                return
            self._installMissingDependencies(missing)
        except Exception as exc:
            self.statusLabel.text = "Dependency installation failed"
            slicer.util.errorDisplay(str(exc))

    def _installMissingDependencies(self, missing):
        if "torch" in missing:
            # Follow KitwareMedical/SlicerNNUnet: PyTorch must be installed by
            # the Slicer PyTorch extension (PyTorchUtils), never by ordinary pip.
            # This keeps the wheel compatible with Slicer's Python and CUDA.
            if not self._installPyTorchWithSlicerExtension():
                return
            missing = self.logic.missingDependencies()
            if "torch" in missing:
                raise RuntimeError(
                    "The PyTorch extension completed but Python still cannot import "
                    "torch. Close every Slicer window, restart Slicer, and press "
                    "Install/verify Python dependencies again."
                )
        if not missing:
            self.statusLabel.text = "All Python dependencies are available"
            slicer.util.infoDisplay("PyTorch is installed and ready.")
            return
        packageDescriptions = [
            f"{module}: {self.logic.REQUIRED_DEPENDENCIES[module]}"
            for module in missing
        ]
        answer = slicer.util.confirmYesNoDisplay(
            "The following packages are missing from Slicer's Python:\n\n"
            + "\n".join(packageDescriptions)
            + "\n\nInstall them now? Slicer must be restarted afterward."
        )
        if not answer:
            raise RuntimeError(
                "Inference was not started because required Python packages are missing: "
                + ", ".join(missing)
            )
        self.statusLabel.text = "Installing missing dependencies..."
        slicer.app.processEvents()
        for module in missing:
            requirement = self.logic.REQUIRED_DEPENDENCIES[module]
            self._appendLog(f"Installing {requirement} ...")
            # Install one requirement at a time. This avoids quoting problems
            # with comparison operators in requirements such as monai>=1.3,<2.
            slicer.util.pip_install(requirement)
        importlib.invalidate_caches()
        stillMissing = self.logic.missingDependencies()
        if stillMissing:
            raise RuntimeError(
                "Installation completed but these imports are still unavailable: "
                + ", ".join(stillMissing)
                + ". Restart Slicer and verify again."
            )
        self.statusLabel.text = "Dependencies installed — restart Slicer"
        slicer.util.infoDisplay(
            "Dependencies were installed successfully. Close all Slicer windows, "
            "restart Slicer, and then run segmentation."
        )

    def _installPyTorchWithSlicerExtension(self):
        """Install Slicer's PyTorch extension/package using its supported API.

        Returns True only when ``torch`` is importable in the current Slicer
        process. Returns False after installing the extension itself because a
        Slicer restart is required before PyTorchUtils can be imported.
        """
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            extensionManager = slicer.app.extensionsManagerModel()
            extensionName = "PyTorch"
            if extensionManager.isExtensionInstalled(extensionName):
                self.statusLabel.text = "Restart Slicer to activate PyTorch"
                slicer.util.infoDisplay(
                    "The PyTorch extension is installed but is not active in this "
                    "Slicer session. Close every Slicer window, restart Slicer, "
                    "then press Install/verify Python dependencies again."
                )
                return False

            answer = slicer.util.confirmYesNoDisplay(
                "The Slicer PyTorch extension is required but is not installed.\n\n"
                "Install it automatically from Slicer's Extension Server now?\n\n"
                "Slicer must be restarted after installation."
            )
            if not answer:
                raise RuntimeError(
                    "PyTorch installation was canceled. Install the PyTorch "
                    "extension from Extension Manager before running this module."
                )
            self.statusLabel.text = "Installing Slicer PyTorch extension..."
            slicer.app.processEvents()
            if not extensionManager.installExtensionFromServer(extensionName):
                raise RuntimeError(
                    "Slicer could not install the PyTorch extension from its server. "
                    "Open Extension Manager, install PyTorch manually, and restart "
                    "Slicer."
                )
            self.statusLabel.text = "PyTorch extension installed — restart Slicer"
            slicer.util.infoDisplay(
                "The PyTorch extension was installed successfully. Close every "
                "Slicer window and restart Slicer. Then return to this module and "
                "press Install/verify Python dependencies once more."
            )
            return False

        answer = slicer.util.confirmYesNoDisplay(
            "The Slicer PyTorch extension is available, but its PyTorch Python "
            "package is not installed.\n\nInstall the compatible PyTorch package now? "
            "This can take several minutes."
        )
        if not answer:
            raise RuntimeError("PyTorch package installation was canceled.")

        self.statusLabel.text = "Installing compatible PyTorch package..."
        self._appendLog("Installing PyTorch through Slicer's PyTorchUtils...")
        slicer.app.processEvents()
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        try:
            result = torchLogic.installTorch(askConfirmation=False)
        except TypeError:
            # Compatibility fallback for older PyTorch extension releases.
            result = torchLogic.installTorch()
        if result is None:
            raise RuntimeError(
                "The Slicer PyTorch extension could not install PyTorch. Review "
                "Slicer's Python console/error log and retry from Extension Manager."
            )

        importlib.invalidate_caches()
        if importlib.util.find_spec("torch") is None:
            self.statusLabel.text = "PyTorch installed — restart Slicer"
            slicer.util.infoDisplay(
                "PyTorch was installed, but it is not visible to the current Python "
                "session yet. Close every Slicer window, restart Slicer, and verify "
                "dependencies again."
            )
            return False

        self._appendLog("PyTorch installation verified.")
        return True

    def _run(self):
        try:
            self._validateInputs()
            missing = self.logic.missingDependencies()
            if missing:
                # Dependency validation happens before CT export and before
                # nnU-Net, preventing a late Stage-2 failure after a long run.
                self._installMissingDependencies(missing)
                return
            self.logText.clear()
            self._temporaryDirectory = qt.QTemporaryDir()
            if not self._temporaryDirectory.isValid():
                raise RuntimeError("Unable to create a temporary inference directory")
            work = Path(self._temporaryDirectory.path())
            batchMode = self.inputModeComboBox.currentIndex == 1
            saveDirectory = Path(self.outputDirectory.currentPath).expanduser().resolve()
            saveDirectory.mkdir(parents=True, exist_ok=True)
            worker = Path(__file__).resolve().parent / "Resources" / "Python" / "ccir_inference.py"
            python = self.logic.findPythonSlicer()
            args = [
                str(worker),
                "--nnunet-model-dir", self.modelDirectory.currentPath,
                "--ccir-checkpoint", self.ccirCheckpoint.currentPath,
                "--configuration", self.configurationEdit.text.strip(),
                "--trainer", self.trainerEdit.text.strip(),
                "--plans", self.plansEdit.text.strip(),
                "--folds", self.foldsEdit.text.strip(),
                "--nnunet-checkpoint", self.nnUNetCheckpointEdit.text.strip(),
                "--component-volume-threshold-ml", str(self.volumeGateSpinBox.value),
                "--remove-threshold", str(self.removeThresholdSpinBox.value),
                "--keep-refine-threshold", str(self.keepThresholdSpinBox.value),
                "--gpu", str(self.gpuSpinBox.value),
                "--npp", str(self.nppSpinBox.value),
                "--nps", str(self.npsSpinBox.value),
            ]
            if batchMode:
                args.extend([
                    "--input-folder", self.inputDirectory.currentPath,
                    "--output-folder", str(saveDirectory),
                ])
                self._expectedOutputPath = None
                self._reportPath = None
                self._batchSummaryPath = saveDirectory / "orbit_ccir_batch_summary.json"
            else:
                inputDirectory = work / "input"
                inputDirectory.mkdir(parents=True, exist_ok=True)
                inputPath = inputDirectory / "slicer_case_0000.nii.gz"
                inputNode = self.inputSelector.currentNode()
                if not slicer.util.exportNode(inputNode, str(inputPath)):
                    raise RuntimeError(f"Failed to export the selected CT to {inputPath}")
                caseName = self.logic.safeCaseName(inputNode.GetName())
                finalPath = saveDirectory / f"{caseName}_seg.nii.gz"
                reportPath = saveDirectory / f"{caseName}_ccir_report.json"
                if finalPath.exists() and not self.overwriteCheckBox.checked:
                    raise FileExistsError(
                        f"Output already exists: {finalPath}. Enable overwrite or "
                        "select another output folder."
                    )
                args.extend([
                    "--input", str(inputPath),
                    "--output", str(finalPath),
                    "--report", str(reportPath),
                    "--case-id", caseName,
                ])
                self._expectedOutputPath = finalPath
                self._reportPath = reportPath
                self._batchSummaryPath = None
            if self.overwriteCheckBox.checked:
                args.append("--overwrite")
            if self.disableResidualCheckBox.checked:
                args.append("--disable-residual-addition")
            if self.disableTtaCheckBox.checked:
                args.append("--disable-tta")

            self._batchMode = batchMode
            self._process = qt.QProcess()
            self._process.setProcessChannelMode(qt.QProcess.MergedChannels)
            # With MergedChannels, readyRead/readAll is portable across the
            # PythonQt versions bundled with different Slicer releases.
            self._process.readyRead.connect(self._readProcessOutput)
            self._process.finished.connect(self._onProcessFinished)
            self._process.errorOccurred.connect(self._onProcessError)
            self.statusLabel.text = "Running ORBIT-RSNA and CCIR..."
            self.applyButton.enabled = False
            self.cancelButton.enabled = True
            self._appendLog(f"Worker: {python}")
            self._appendLog(f"Temporary workspace: {work}")
            self._appendLog(f"Output folder: {saveDirectory}")
            self._process.start(str(python), args)
        except Exception as exc:
            self._process = None
            self.statusLabel.text = "Could not start"
            self.cancelButton.enabled = False
            self._updateApplyState()
            slicer.util.errorDisplay(str(exc))

    def _readProcessOutput(self):
        if self._process is None:
            return
        stream = self._process.readAll()
        # QByteArray in Slicer's PythonQt binding is not necessarily accepted
        # by Python's bytes() constructor. QTextCodec is the conversion method
        # used by SlicerNNUnet and works with PythonQt QByteArray directly.
        data = qt.QTextCodec.codecForUtfText(stream).toUnicode(stream)
        self._appendLog(data)

    def _onProcessError(self, error):
        self._appendLog(f"Process error: {error}")

    def _onProcessFinished(self, exitCode=0, exitStatus=None):
        # Depending on Slicer's Qt/PythonQt version, QProcess.finished may be
        # exposed as finished(int) or finished(int, ExitStatus). The optional
        # second argument supports both bindings.
        self._readProcessOutput()
        process = self._process
        self._process = None
        self.cancelButton.enabled = False
        try:
            if int(exitCode) != 0:
                if (
                    getattr(self, "_batchMode", False)
                    and self._batchSummaryPath is not None
                    and self._batchSummaryPath.is_file()
                ):
                    summary = json.loads(
                        self._batchSummaryPath.read_text(encoding="utf-8")
                    )
                    self._appendLog(
                        "\nPartial batch summary:\n" + json.dumps(summary, indent=2)
                    )
                raise RuntimeError(
                    f"Inference failed with exit code {exitCode}. Review the log above."
                )
            if self._batchMode:
                if not self._batchSummaryPath.is_file():
                    raise RuntimeError("Batch finished without creating its summary JSON")
                summary = json.loads(
                    self._batchSummaryPath.read_text(encoding="utf-8")
                )
                self._appendLog("\nBatch summary:\n" + json.dumps(summary, indent=2))
                self.statusLabel.text = (
                    f"Completed: {summary.get('completed', 0)} cases; "
                    f"skipped: {summary.get('skipped', 0)}"
                )
                slicer.util.showStatusMessage(
                    "ORBIT-RSNA + CCIR batch segmentation completed", 5000
                )
                return
            if self._expectedOutputPath is None or not self._expectedOutputPath.is_file():
                raise RuntimeError("Inference finished without creating the segmentation")
            segmentationNode = slicer.util.loadSegmentation(str(self._expectedOutputPath))
            if segmentationNode is None:
                raise RuntimeError("Inference succeeded but Slicer could not load the segmentation")
            segmentationNode.SetName(
                slicer.mrmlScene.GenerateUniqueName(
                    f"{self.inputSelector.currentNode().GetName()}_ORBIT_CCIR"
                )
            )
            self.logic.nameAndColorSegments(segmentationNode)
            if self._reportPath.is_file():
                report = json.loads(self._reportPath.read_text(encoding="utf-8"))
                self._appendLog("\nFinal report:\n" + json.dumps(report, indent=2))
            self.statusLabel.text = "Completed"
            slicer.util.setSliceViewerLayers(background=self.inputSelector.currentNode())
            slicer.util.showStatusMessage(
                "ORBIT-RSNA + CCIR segmentation completed", 5000
            )
        except Exception as exc:
            self.statusLabel.text = "Failed"
            slicer.util.errorDisplay(str(exc))
        finally:
            if process is not None:
                process.deleteLater()
            self._updateApplyState()

    def _cancel(self):
        if self._process is not None:
            self._appendLog("Cancelling inference...")
            self._process.kill()
            self._process.waitForFinished(5000)
            self._process.deleteLater()
            self._process = None
        if hasattr(self, "cancelButton"):
            self.cancelButton.enabled = False
        if hasattr(self, "statusLabel"):
            self.statusLabel.text = "Cancelled"
        self._updateApplyState()

    def _validateInputs(self):
        batchMode = self.inputModeComboBox.currentIndex == 1
        if batchMode:
            inputDirectoryText = self.inputDirectory.currentPath.strip()
            if not inputDirectoryText:
                raise ValueError("Select an input NIfTI folder")
            inputDirectory = Path(inputDirectoryText)
            if not inputDirectory.is_dir():
                raise FileNotFoundError(f"Input folder not found: {inputDirectory}")
        elif self.inputSelector.currentNode() is None:
            raise ValueError("Select an input NCCT volume")
        outputDirectoryText = self.outputDirectory.currentPath.strip()
        if not outputDirectoryText:
            raise ValueError("Select an output save folder")
        outputDirectory = Path(outputDirectoryText)
        if not outputDirectory.is_dir():
            raise FileNotFoundError(f"Output folder not found: {outputDirectory}")
        modelDirectoryText = self.modelDirectory.currentPath.strip()
        if not modelDirectoryText:
            raise ValueError("Select the method2 ORBIT Dataset folder")
        modelDirectory = Path(modelDirectoryText)
        if not modelDirectory.is_dir():
            raise FileNotFoundError(f"nnU-Net Dataset folder not found: {modelDirectory}")
        hasDatasetJson = (modelDirectory / "dataset.json").is_file() or any(
            modelDirectory.glob("*/dataset.json")
        )
        if not hasDatasetJson:
            raise FileNotFoundError(
                f"dataset.json is missing from {modelDirectory} and its immediate "
                "configuration folders"
            )
        checkpointText = self.ccirCheckpoint.currentPath.strip()
        if not checkpointText:
            raise ValueError("Select the CCIR checkpoint")
        ccirCheckpoint = Path(checkpointText)
        if not ccirCheckpoint.is_file():
            raise FileNotFoundError(f"CCIR checkpoint not found: {ccirCheckpoint}")
        if self.keepThresholdSpinBox.value < self.removeThresholdSpinBox.value:
            raise ValueError("Keep/refine probability must be >= remove probability")
        if self.trainerEdit.text.strip() != "nnUNetTrainerORBITRSNA":
            raise ValueError(
                "Trainer must be nnUNetTrainerORBITRSNA for the method2 baseline"
            )
        if self.plansEdit.text.strip() != "nnUNetResEncUNetLPlans":
            raise ValueError(
                "Plans must be nnUNetResEncUNetLPlans for the method2 baseline"
            )


class ICH4LabelCCIRLogic(ScriptedLoadableModuleLogic):
    REQUIRED_DEPENDENCIES = {
        # Torch must be provided by Slicer's PyTorch extension, not regular pip.
        "torch": "Slicer PyTorch extension",
        "nnunetv2": "nnunetv2==2.8.1",
        "dynamic_network_architectures": "dynamic-network-architectures==0.4.4",
        "monai": "monai>=1.3,<2",
        "nibabel": "nibabel>=5.1",
        "scipy": "scipy>=1.10",
        "pandas": "pandas>=2.0",
    }

    @classmethod
    def missingDependencies(cls):
        return [
            module
            for module in cls.REQUIRED_DEPENDENCIES
            if importlib.util.find_spec(module) is None
        ]

    @staticmethod
    def safeCaseName(name: str) -> str:
        cleaned = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in str(name).strip()
        ).strip("_")
        return cleaned or "slicer_case"

    @staticmethod
    def findPythonSlicer() -> Path:
        """Find PythonSlicer for an isolated worker with Slicer's packages/environment."""
        names = ("PythonSlicer.exe", "PythonSlicer")
        appDirectory = Path(slicer.app.applicationDirPath()).resolve()
        roots = [
            appDirectory,
            appDirectory / "bin",
            appDirectory.parent,
            appDirectory.parent / "bin",
            appDirectory.parent / "MacOS",
        ]
        for root in roots:
            for name in names:
                candidate = root / name
                if candidate.is_file():
                    return candidate
        onPath = shutil.which("PythonSlicer") or shutil.which("PythonSlicer.exe")
        if onPath:
            return Path(onPath)
        if "python" in Path(sys.executable).name.lower():
            return Path(sys.executable)
        raise RuntimeError(
            "Could not find PythonSlicer. Run this module inside a standard 3D Slicer "
            "installation and verify that PythonSlicer exists in the application bin folder."
        )

    @staticmethod
    def nameAndColorSegments(segmentationNode):
        labels = {
            1: ("ICH", (0.85, 0.10, 0.10)),
            2: ("Edema", (0.95, 0.75, 0.10)),
            3: ("Ventricle", (0.10, 0.55, 0.95)),
            4: ("IVH", (0.85, 0.15, 0.85)),
        }
        segmentation = segmentationNode.GetSegmentation()
        for index in range(segmentation.GetNumberOfSegments()):
            segmentId = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(segmentId)
            labelValue = None
            candidates = (segmentId, segment.GetName())
            for candidate in candidates:
                digits = "".join(character for character in str(candidate) if character.isdigit())
                if digits:
                    parsed = int(digits)
                    if parsed in labels:
                        labelValue = parsed
                        break
            if labelValue is None:
                # Plain labelmap imports normally preserve ascending label order.
                labelValue = index + 1
            if labelValue in labels:
                name, color = labels[labelValue]
                segment.SetName(name)
                segment.SetColor(*color)


class ICH4LabelCCIRTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("ICH4LabelCCIR module loaded successfully")
        self.assertTrue((Path(__file__).resolve().parent / "Resources" / "Python" / "ccir_inference.py").is_file())
