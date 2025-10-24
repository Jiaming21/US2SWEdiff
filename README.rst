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

Inference
=============

Step 1: Prepare Conda Environment
======================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.  
Then, create environment and install packages and dependencies using the following command (here CUDA 11.3):

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

*(This step is only required if you wish to apply the model to your own dataset.  
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

Each subfolder under ``infer/`` should contain your ultrasound (US) images in standard format (e.g., ``.png``, ``.jpg``, or ``.tif``).

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

    .. code-block:: bash

       ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

    .. note::

       - On the first connection, if prompted with *yes/no*, type ``yes``.  
       - Enter the server password (it will not be displayed while typing or pasting — this is normal).  
       - If you see ``Permission denied``, the password was likely incorrect. Please retry.

    After connecting, open your browser at: ``http://localhost:6006`` to access the Gradio interface.

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~

    On your *local terminal* (PowerShell for Windows, or Terminal for macOS/Linux):

    .. code-block:: bash

       cd ControlNet-main/gradio
       python app.py

    Once the Gradio server has started, the terminal will display something like:

    .. code-block:: text

       Running on local URL:  http://127.0.0.1:7860/

    Now open your browser and go to the displayed URL (commonly ``http://127.0.0.1:7860`` or ``http://localhost:7860``).

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. **Upload an image**: Click the top-left window to upload your input image.  
2. **Enter the prompt**: In the *prompt* field, type your description, e.g.:  
   ``a photo of a benign breast tumor`` or ``a photo of a malignant breast tumor``.  
3. **Generate**: Click **Generate**. The right-hand panel will display  
   the extracted **Laplacian edge** and the generated **SWE image**.

**Advanced options:**
    - **Images** — the number of images to generate.
    - **Laplacian ksize (odd)** — kernel size for the Laplacian edge detector (odd integers only: 1, 3, 5, 7, …).

Option 2: Using Provided Scripts
------------------------------------

In the following example, we demonstrate inference using the best-performing model  
(*Laplacian edge map → SWE image*) on the **BUSI** dataset.

Step 1–3: Repeat Previous Instructions
===========================

Repeat **Step 1–3** from the *Inference* section to set up the environment, clone the repository, and prepare the dataset.

Step 4: Create the "metadata.json" File
===========================

.. code-block:: bash

    cd [your_path_to_ControlNet-main_folder]/data/tools/

Modify the ``data.py`` file under this directory and ensure the paths are correct:

.. code-block:: python

    imagepath = "../infer/BUSI/*"  # arbitrary, since inference doesn't require label images
    condpath = "../infer/laplacian/"  # path to condition images (e.g., Laplacian edges)
    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/"  # dataset root

    with open("../infer/metadata.json", 'w') as f:  # output JSON file name

After verifying the settings, run:

.. code-block:: bash

    python data.py

This will create ``metadata.json`` under the ``../infer/`` folder.

Step 5: Build the Inference Dataset
===========================

Open ``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py``  
and modify:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/metadata.json"

Step 6: Load the ControlNet Model
===========================

Your trained model checkpoints reside under ``ControlNet-main/lightning_logs/``.  
For example:

.. code-block:: text

    [your_path]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt

Open ``[your_path_to_ControlNet-main_folder]/tutorial_inference.py``  
and modify the following lines:

.. code-block:: python

    CKPT_PATH = "[your_path]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt"
    RESULT_DIR = "[your_path]/generated_results/"

Then run:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/tutorial_inference.py

Results will be saved under:

.. code-block:: text

    [your_path_to_ControlNet-main_folder]/generated_results/version_0/

Train
=============

In this section, we train the best-performing model (*Laplacian edge map → SWE image*).

Step 1–2: Prepare Conda Environment & Pull from GitHub Repository
===========================

Repeat **Step 1** and **Step 2** from the *Inference* section.

Step 3: Prepare the Dataset
===========================

*(This step is only required if you wish to train the model on your own dataset.  
For this project, all data are already well organized when you clone the repository.)*

.. code-block:: text

    Breast-img/
    └── Train/
        ├── us/
        ├── canny/
        ├── laplacian/   # used as condition images
        └── swe/         # used as target images

Step 4: Create the "metadata.json" File
===========================

.. code-block:: bash

    cd [your_path_to_ControlNet-main_folder]/data/tools/

Modify ``data.py``:

.. code-block:: python

    imagepath = "../train/swe/"          # path to target images
    condpath = "../train/laplacian/"     # path to condition images
    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/"
    with open("../train/metadata.json", 'w') as f:

Then run:

.. code-block:: bash

    python data.py

Step 5: Build the Training Dataset
===========================

Open ``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py`` and modify:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/metadata.json"

Step 6: Create Complete Model Weights
===========================

Combine Stable Diffusion + ControlNet weights:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py \
    [your_path]/models/stable-diffusion-v1-5/v1-5-pruned.ckpt \
    [your_path]/models/stable-diffusion-v1-5/controlnet.ckpt

Step 7: Load and Train the Model
===========================

.. code-block:: python

    resume_path = '[your_path]/models/stable-diffusion-v1-5/controlnet.ckpt'

Train with:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tutorial_train.py

**Training results:**

1. **Model checkpoints** — saved under:
   ``lightning_logs/version_1/checkpoints/``  
2. **Visualization logs** — stored in  
   ``image_log/train/`` and include:
   - Conditioning (prompt)  
   - Control (Laplacian edge map)  
   - Reconstruction (true SWE images)  
   - Samples (synthesized SWE images)

Advanced Options for Training
=============
1. Improved Hint Block  
2. Unlocked Decoder  
3. Classifier-free Guidance


