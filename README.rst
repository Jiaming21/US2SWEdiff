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

* `Inference <#inference>`_

    * `Step 1: Prepare Conda Environment <#step-1-prepare-conda-environment>`_
    * `Step 2: Pull from GitHub Repository <#step-2-pull-from-github-repository>`_
    * `Step 3: Prepare the Dataset <#step-3-prepare-the-dataset>`_
    
	* `Option 1: Using Gradio Interface <#option-1-using-gradio-interface>`_

        	* `Run on Remote Server <#run-on-remote-server>`_
		* `Run on Local Computer <#run-on-local-computer>`_
		* `Gradio Interface Usage Instructions <#gradio-interface-usage-instructions>`_

        * `Option 2: Using Provided Scripts <#option-2-using-provided-scripts>`_

    		* `Step 1–3: Repeat Previous Instructions <#step-13-repeat-previous-instructions>`_
    		* `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file>`_
    		* `Step 5: Build the Inference Dataset <#step-5-build-the-inference-dataset>`_
    		* `Step 6: Load the ControlNet Model <#step-6-load-the-controlnet-model>`_

* `Train <#train>`_

  * `Step 1–2: Prepare Conda Environment & Pull from GitHub Repository <#step-1-2-prepare-conda-environment--pull-from-github-repository>`_
  * `Step 3: Prepare the Dataset <#step-3-prepare-the-dataset>`_
  * `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file-train>`_
  * `Step 5: Build the Training Dataset <#step-5-build-the-training-dataset>`_
  * `Step 6: Create Complete Model Weights <#step-6-create-complete-model-weights>`_
  * `Step 7: Load and Train the Model <#step-7-load-and-train-the-model>`_

---

.. _inference:

Inference
=============

.. _step-1-prepare-conda-environment:

Step 1: Prepare Conda Environment
=================================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code-block:: bash

    conda create -n controlnet python=3.10
    conda activate controlnet
    conda env update -n controlnet -f controlnet.yaml


.. _step-2-pull-from-github-repository:

Step 2: Pull from GitHub Repository
===================================

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff


Model Files
===========

Model files are hosted on 🤗 Hugging Face due to size and license constraints.

- https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5
- https://huggingface.co/Jiaming2143183/clip-vit-large-patch14


.. _step-3-prepare-the-dataset:

Step 3: Prepare the Dataset
===========================

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


.. _option-1-using-gradio-interface:

Option 1: Using Gradio Interface
--------------------------------

.. _run-on-remote-server:

**Run on Remote Server**
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py

Then, on your local computer:

.. code-block:: bash

    ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

Open your browser at ``http://localhost:6006``


.. _run-on-local-computer:

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd ControlNet-main/gradio
    python app.py


.. _gradio-interface-usage-instructions:

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. Upload your input image  
2. Enter a text prompt  
3. Click **Generate**

Advanced options:
- **Images** — number of outputs  
- **Laplacian ksize** — edge kernel size


.. _option-2-using-provided-scripts:

Option 2: Using Provided Scripts
--------------------------------

.. _step-13-repeat-previous-instructions:

Step 1–3: Repeat Previous Instructions
======================================
Repeat Step 1–3 from *Inference*.


.. _step-4-create-the-metadatajson-file:

Step 4: Create the "metadata.json" File
=======================================

Modify ``data.py``:

.. code-block:: python

    imagepath = "../infer/BUSI/*"
    condpath  = "../infer/laplacian/"
    root      = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/"

Run:

.. code-block:: bash

    python data.py


.. _step-5-build-the-inference-dataset:

Step 5: Build the Inference Dataset
===================================

Edit ``tutorial_dataset.py``:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/metadata.json"


.. _step-6-load-the-controlnet-model:

Step 6: Load the ControlNet Model
=================================

.. code-block:: python

    CKPT_PATH = "[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt"
    RESULT_DIR = "[your_path_to_ControlNet-main_folder]/generated_results/"

Run:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/tutorial_inference.py

Output directory:  
``[your_path_to_ControlNet-main_folder]/generated_results/version_0/``


---

.. _train:

Train
=============

.. _step-1-2-prepare-conda-environment--pull-from-github-repository:

Step 1–2: Prepare Conda Environment & Pull from GitHub Repository
=================================================================
Repeat Step 1–2 from *Inference*.


.. _step-3-prepare-the-dataset-train:

Step 3: Prepare the Dataset
===========================

.. code-block:: text

    Breast-img/
    └── Train/
        ├── us/
        ├── canny/
        ├── laplacian/
        └── swe/


.. _step-4-create-the-metadatajson-file-train:

Step 4: Create the "metadata.json" File
=======================================

Modify ``data.py``:

.. code-block:: python

    imagepath = "../train/swe/"
    condpath  = "../train/laplacian/"
    root      = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/"

Run:

.. code-block:: bash

    python data.py


.. _step-5-build-the-training-dataset:

Step 5: Build the Training Dataset
==================================

Edit ``tutorial_dataset.py``:

.. code-block:: python

    root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/metadata.json"


.. _step-6-create-complete-model-weights:

Step 6: Create Complete Model Weights
=====================================

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py \
      [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/v1-5-pruned.ckpt \
      [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/controlnet.ckpt

This creates ``controlnet.ckpt`` (SD + ControlNet combined weights).


.. _step-7-load-and-train-the-model:

Step 7: Load and Train the Model
================================

.. code-block:: python

    resume_path = "[your_path_to_ControlNet-main_folder]/models/stable-diffusion-v1-5/controlnet.ckpt"

Run:

.. code-block:: bash

    python [your_path_to_ControlNet-main_folder]/ControlNet-main/tutorial_train.py


Training results:
-----------------

1. **Model checkpoints** — saved in ``lightning_logs/version_1/checkpoints/``  
2. **Visualization logs** — stored in ``image_log/train/`` and include:
   - Conditioning (prompt)
   - Control (Laplacian edge map)
   - Reconstruction (true SWE images)
   - Samples (synthesized SWE images)