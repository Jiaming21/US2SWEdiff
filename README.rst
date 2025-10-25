=========
US2SWEdiff
=========
US2SWEdiff is a tool which generates SWE images corresponding to US images using diffusion model.

Overview
=============

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/US2SWEdiff_logo.png
   :width: 180

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/model.jpg
   :width: 1000


Quick Start
=============

Use the following links to quickly navigate through the documentation:

- `Inference <#inference>`_
  - `Option 1: Using Gradio Interface <#option-1-using-gradio-interface>`_
    - `Run on Remote Server <#run-on-remote-server>`_
    - `Run on Local Computer <#run-on-local-computer>`_
    - `Gradio Interface Usage Instructions <#gradio-interface-usage-instructions>`_
  - `Option 2: Using Provided Scripts <#option-2-using-provided-scripts>`_
    - `Step 1–3: Repeat Previous Instructions <#step-1-3-repeat-previous-instructions>`_
    - `Step 4: Create the "metadata.json" File <#step-4-create-your-metadatajson-file>`_
    - `Step 5: Build the Inference Dataset <#step-5-build-the-dataset-for-inferrence-by-using-the-metadatajson-file>`_
    - `Step 6: Load the ControlNet Model <#step-6-load-your-controlnet-model-refer-to-cldmcldmpy-with-previously-trained-weights>`_

- `Train <#train>`_
  - `Step 1–2: Prepare Conda Environment & Pull from GitHub Repository <#step-1-2-prepare-conda-environment--pull-from-github-repository>`_
  - `Step 3: Prepare the Dataset <#step-3-prepare-the-dataset>`_
  - `Step 4: Create the "metadata.json" File <#step-4-create-your-metadatajson-file>`_
  - `Step 5: Build the Training Dataset <#step-5-build-the-dataset-for-training-by-using-the-metadatajson-file>`_
  - `Step 6: Create Complete Model Weights <#step-6-make-complete-model-parameters>`_
  - `Step 7: Load and Train the Model <#step-7-load-and-train-the-model>`_

---

Inference
=============

Step 1: Prepare Conda Environment
======================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. Then, create environment and install packages and dependencies using following command (here CUDA 11.3):

.. code-block:: bash

    # Create a new environment named "controlnet" with Python 3.10
    conda create -n controlnet python=3.10

    # Activate the environment
    conda activate controlnet

    # Install dependencies from controlnet.yaml (environment reproduction)
    conda env update -n controlnet -f controlnet.yaml

This will create a conda environment named ``controlnet`` with packages and dependencies installed.

Step 2: Pull from GitHub Repository
======================
Clone the US2SWEdiff repository from GitHub:

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff

Model Files
===========

The large model files used in this project (``stable-diffusion-v1-5`` and ``clip-vit-large-patch14``)
are stored separately on the 🤗 Hugging Face Hub for size and licensing reasons.

For more information about these models and their usage conditions, please refer to:

``models/model_files_notice.txt``

Or visit the model pages directly:

- Stable Diffusion v1.5: https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5  
- CLIP ViT-L/14: https://huggingface.co/Jiaming2143183/clip-vit-large-patch14


Step 3: Prepare the Dataset
===========================

*(This step is only required if you wish to apply the model infer your own dataset.  
For this project, all data are already well organized when you clone the repository.)*

The dataset directory structure should look like this:

.. code-block:: text

    Breast-img/
    └── infer/
        ├── BLUSG/
        │   ├── canny/
        │   ├── laplacian/
        │   └── us/
        ├── BUSBRA/
        │   ├── canny/
        │   ├── laplacian/
        │   └── us/
        ├── BUSI/
        │   ├── canny/
        │   ├── laplacian/
        │   └── us/
        └── your_dataset/
            ├── canny/
            ├── laplacian/
            └── us/

Each subfolder under ``Infer/`` should contain your ultrasound (US) images in standard format (e.g., ``.png``, ``.jpg``, or ``.tif``).

Step 4: Run Inference
======================

After completing the environment setup and cloning the repository (see Step 1 and Step 2), 
you can perform inference using either the **Gradio** graphical interface or command line.

.. contents::
   :local:
   :depth: 2

Option 1: Using Gradio Interface
------------------------------------

You can run the Gradio interface in **two ways**:

1. On a **remote server** with SSH port forwarding.
2. Directly on your **local computer**.

**Run on Remote Server**
~~~~~~~~~~~~~~~~~~~

    On the *remote server* (Linux terminal):

    .. code-block:: bash

       cd ControlNet-main/gradio
       python app.py

    On your *local machine*, establish SSH port forwarding:

    - **Windows**: open *PowerShell*
    - **macOS / Linux**: open *Terminal*

    .. code-block:: bash

       ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~

    .. code-block:: bash

       cd ControlNet-main/gradio
       python app.py

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. **Upload an image**  
2. **Enter the prompt** (e.g., ``a photo of a benign breast tumor``)  
3. **Generate** to produce SWE images.  

**Advanced options:**
    - **Images** — number of generated images  
    - **Laplacian ksize (odd)** — kernel size of edge detector  

Option 2: Using Provided Scripts
------------------------------------

Step 1–3: Repeat Previous Instructions
===========================

Repeat **Step 1–3** from the *Inference* section to set up the environment, clone the repository and prepare the dataset.

Step 4: Create the "metadata.json" File
===========================

*(Modify `data.py` under `[your_path_to_ControlNet-main_folder]/data/tools/` as shown.)*

Step 5: Build the Inference Dataset
===========================

*(Modify `tutorial_dataset.py` to use your generated metadata file.)*

Step 6: Load the ControlNet Model
===========================

*(Edit `tutorial_inference.py` to set correct checkpoint and result paths, then run inference.)*


Train
=============

Step 1–2: Prepare Conda Environment & Pull from GitHub Repository
===========================

Repeat **Step 1** and **Step 2** from the *Inference* section.

Step 3: Prepare the Dataset
===========================

*(Same logic as in Inference but for the training dataset.)*

Step 4: Create the "metadata.json" File
===========================

*(Modify and run `data.py` under `data/tools/` for training.)*

Step 5: Build the Training Dataset
===========================

*(Edit `tutorial_dataset.py` for training mode.)*

Step 6: Create Complete Model Weights
===========================

*(Run `tool_add_control.py` to combine SD + ControlNet parameters.)*

Step 7: Load and Train the Model
===========================

.. code-block:: bash

    resume_path = '[your_path_to_ControlNet-main_folder]/models/stable-diffusion-v1-5/controlnet.ckpt'

Train with:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tutorial_train.py

Training results:
-----------------
1. **Model checkpoints** — saved under ``lightning_logs/version_1/checkpoints/``  
2. **Visualization logs** — stored in ``image_log/train/`` and include:  
   - Conditioning (prompt)  
   - Control (Laplacian edge map)  
   - Reconstruction (true SWE images)  
   - Samples (synthesized SWE images)

---