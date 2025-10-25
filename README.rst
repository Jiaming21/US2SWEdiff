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
    - `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file>`_
    - `Step 5: Build the Inference Dataset <#step-5-build-the-inference-dataset>`_
    - `Step 6: Load the ControlNet Model <#step-6-load-the-controlnet-model>`_

- `Train <#train>`_
  - `Step 1–2: Prepare Conda Environment & Pull from GitHub Repository <#step-1-2-prepare-conda-environment--pull-from-github-repository>`_
  - `Step 3: Prepare the Dataset <#step-3-prepare-the-dataset>`_
  - `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file-1>`_
  - `Step 5: Build the Training Dataset <#step-5-build-the-training-dataset>`_
  - `Step 6: Create Complete Model Weights <#step-6-create-complete-model-weights>`_
  - `Step 7: Load and Train the Model <#step-7-load-and-train-the-model>`_

---

Inference
=============

Step 1: Prepare Conda Environment
======================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.  
Then create an environment and install dependencies using:

.. code-block:: bash

    conda create -n controlnet python=3.10
    conda activate controlnet
    conda env update -n controlnet -f controlnet.yaml

This creates a conda environment named ``controlnet`` with all required packages.

Step 2: Pull from GitHub Repository
======================
Clone the US2SWEdiff repository from GitHub:

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff

Model Files
===========
The model files (``stable-diffusion-v1-5`` and ``clip-vit-large-patch14``)  
are hosted on 🤗 Hugging Face for licensing and size reasons.

See ``models/model_files_notice.txt`` or visit:
- https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5  
- https://huggingface.co/Jiaming2143183/clip-vit-large-patch14

Step 3: Prepare the Dataset
===========================

*(This step is only required if you wish to apply the model to your own dataset.  
For this project, all data are already well organized when you clone the repository.)*

Dataset structure:

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

Each subfolder under ``infer/`` should contain ultrasound (US) images in standard formats (``.png``, ``.jpg``, ``.tif``).

Step 4: Run Inference
======================
After completing environment setup and cloning the repo,  
you can perform inference via **Gradio** interface or command line.

.. contents::
   :local:
   :depth: 2

Option 1: Using Gradio Interface
------------------------------------

You can run the Gradio interface either on a **remote server** or **locally**.

**Run on Remote Server**
~~~~~~~~~~~~~~~~~~~
On the *remote server*:

.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py

Then on your *local machine* (PowerShell or Terminal):

.. code-block:: bash

    ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

Open your browser at ``http://localhost:6006``.

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py

Then open your browser at the displayed local URL (e.g., ``http://127.0.0.1:7860``).

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. **Upload an image**  
2. **Enter the prompt**, e.g. ``a photo of a benign breast tumor``  
3. **Click Generate** — SWE images will appear in the output panel.

**Advanced options:**
- **Images** — number of images to generate  
- **Laplacian ksize (odd)** — edge detector kernel size (1, 3, 5, …)

Option 2: Using Provided Scripts
------------------------------------

Step 1–3: Repeat Previous Instructions
===========================
Repeat **Step 1–3** from the *Inference* section to set up environment, clone the repo, and prepare the dataset.

Step 4: Create the "metadata.json" File
===========================
Modify ``data.py`` under:
``[your_path_to_ControlNet-main_folder]/data/tools/``

Example:
.. code-block:: python

    imagepath = "../infer/BUSI/*"
    condpath = "../infer/laplacian/"
    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/"

    with open("../infer/metadata.json", 'w') as f:
        ...

Then run:
.. code-block:: bash

    python data.py

This creates the file ``../infer/metadata.json``.

Step 5: Build the Inference Dataset
===========================
Open:
``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py``

Edit:
.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/metadata.json"

Step 6: Load the ControlNet Model
===========================
Use your trained model weights under:
``[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/``

For example:
.. code-block:: python

    CKPT_PATH = "[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt"
    RESULT_DIR = "[your_path_to_ControlNet-main_folder]/generated_results/"

Run inference:
.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/tutorial_inference.py

Results will be stored under:
``[your_path_to_ControlNet-main_folder]/generated_results/version_0/``


Train
=============

Step 1–2: Prepare Conda Environment & Pull from GitHub Repository
===========================
Repeat **Step 1** and **Step 2** from *Inference*.

Step 3: Prepare the Dataset
===========================
*(Only needed if training on your own data.)*

.. code-block:: text

    Breast-img/
    └── Train/
        ├── us/
        ├── canny/
        ├── laplacian/   (used condition images folder)
        └── swe/         (used target images folder)

Each subfolder under ``Train/`` should contain images in ``.png``, ``.jpg``, or ``.tif``.

Step 4: Create the "metadata.json" File
===========================
Modify ``data.py`` under ``data/tools/``:

.. code-block:: python

    imagepath = "../train/swe/"
    condpath = "../train/laplacian/"
    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/"

Then run:
.. code-block:: bash

    python data.py

This creates ``../train/metadata.json``.

Step 5: Build the Training Dataset
===========================
Edit:
``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py``

Set:
.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/metadata.json"

Step 6: Create Complete Model Weights
===========================
Run:
.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py \
    [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/v1-5-pruned.ckpt \
    [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/controlnet.ckpt

This creates:
``controlnet.ckpt`` (SD + ControlNet combined weights)

Step 7: Load and Train the Model
===========================
.. code-block:: python

    resume_path = "[your_path_to_ControlNet-main_folder]/models/stable-diffusion-v1-5/controlnet.ckpt"

Train the model:
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