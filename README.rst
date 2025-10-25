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

- **Inference**

  - `Option 1: Using Gradio Interface <#option-1>`_

    * `Run on Remote Server <#run-on-remote-server>`_
    * `Run on Local Computer <#run-on-local-computer>`_
    * `Gradio Interface Usage Instructions <#gradio-interface-usage-instructions>`_

  - `Option 2: Using Provided Scripts <#option-2>`_

    * `Step 1–3: Repeat Previous Instructions <#inference-step-1-3>`_
    * `Step 4: Create the "metadata.json" File <#inference-step-4>`_
    * `Step 5: Build the Inference Dataset <#inference-step-5>`_
    * `Step 6: Load the ControlNet Model <#inference-step-6>`_

- **Train**

  * `Step 1–2: Prepare Conda Environment & Pull from GitHub Repository <#train-step-1-2>`_
  * `Step 3: Prepare the Dataset <#train-step-3>`_
  * `Step 4: Create the "metadata.json" File <#train-step-4>`_
  * `Step 5: Build the Training Dataset <#train-step-5>`_
  * `Step 6: Create Complete Model Weights <#train-step-6>`_
  * `Step 7: Load and Train the Model <#train-step-7>`_


---

.. _inference:

Inference
=============

.. _inference-step-1:

Step 1: Prepare Conda Environment
======================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Then, create environment and install dependencies:

.. code-block:: bash

    conda create -n controlnet python=3.10
    conda activate controlnet
    conda env update -n controlnet -f controlnet.yaml


.. _inference-step-2:

Step 2: Pull from GitHub Repository
======================

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff


Model Files
===========

Model files are stored on 🤗 Hugging Face due to size/licensing reasons.  
See ``models/model_files_notice.txt`` or visit:

- https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5
- https://huggingface.co/Jiaming2143183/clip-vit-large-patch14


.. _inference-step-3:

Step 3: Prepare the Dataset
===========================

Dataset directory structure:

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


.. _option-1:

Option 1: Using Gradio Interface
------------------------------------

.. _run-on-remote-server:

**Run on Remote Server**
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py

Then on your local machine:

.. code-block:: bash

    ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

Open: ``http://localhost:6006``


.. _run-on-local-computer:

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py


.. _gradio-interface-usage-instructions:

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. Upload an image  
2. Enter a prompt  
3. Click **Generate**

Advanced options:
- Images — number of generated results  
- Laplacian ksize (odd) — kernel size for edge extraction


.. _option-2:

Option 2: Using Provided Scripts
------------------------------------

.. _inference-step-1-3:

Step 1–3: Repeat Previous Instructions
===========================

Repeat **Step 1–3** from *Inference*.


.. _inference-step-4:

Step 4: Create the "metadata.json" File
===========================

Modify ``data.py`` under ``data/tools/``:

.. code-block:: python

    imagepath = "../infer/BUSI/*"
    condpath  = "../infer/laplacian/"
    root      = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/"

Run:

.. code-block:: bash

    python data.py


.. _inference-step-5:

Step 5: Build the Inference Dataset
===========================

Edit ``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py``:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/metadata.json"


.. _inference-step-6:

Step 6: Load the ControlNet Model
===========================

Example:

.. code-block:: python

    CKPT_PATH = "[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt"
    RESULT_DIR = "[your_path_to_ControlNet-main_folder]/generated_results/"

Run:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/tutorial_inference.py

Results will be saved in:
``[your_path_to_ControlNet-main_folder]/generated_results/version_0/``


---

.. _train:

Train
=============

.. _train-step-1-2:

Step 1–2: Prepare Conda Environment & Pull from GitHub Repository
===========================
Repeat **Step 1** and **Step 2** from *Inference*.


.. _train-step-3:

Step 3: Prepare the Dataset
===========================

.. code-block:: text

    Breast-img/
    └── Train/
        ├── us/
        ├── canny/
        ├── laplacian/
        └── swe/


.. _train-step-4:

Step 4: Create the "metadata.json" File
===========================

Modify ``data.py`` under ``data/tools/``:

.. code-block:: python

    imagepath = "../train/swe/"
    condpath  = "../train/laplacian/"
    root      = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/"

Run:

.. code-block:: bash

    python data.py

This creates ``../train/metadata.json``.


.. _train-step-5:

Step 5: Build the Training Dataset
===========================

Edit ``[your_path_to_ControlNet-main_folder]/tutorial_dataset.py``:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/metadata.json"


.. _train-step-6:

Step 6: Create Complete Model Weights
===========================

Run:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py \
      [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/v1-5-pruned.ckpt \
      [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/controlnet.ckpt

This creates ``controlnet.ckpt`` (SD + ControlNet combined weights).


.. _train-step-7:

Step 7: Load and Train the Model
===========================

.. code-block:: python

    resume_path = "[your_path_to_ControlNet-main_folder]/models/stable-diffusion-v1-5/controlnet.ckpt"

Train with:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tutorial_train.py


Training results:
-----------------

1. **Model checkpoints** — stored under ``lightning_logs/version_1/checkpoints/``

2. **Visualization logs** — stored in ``image_log/train/`` and include:

   - **Conditioning** — prompt
   - **Control** — Laplacian edge map
   - **Reconstruction** — true SWE images
   - **Samples** — synthesized SWE images