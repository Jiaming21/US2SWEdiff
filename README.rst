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
-----------------------
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
-----------------------
Clone the US2SWEdiff repository from GitHub:

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff

Step 3 — Run Inference
======================

After completing the environment setup and cloning the repository (see Step 1 and Step 2), 
you can perform inference using either the **Gradio** graphical interface or command line.

.. contents::
   :local:
   :depth: 2

Option 1: Using the Gradio Interface
------------------------------------

On the *remote server* (Linux terminal), start the application:

.. code-block:: bash

   cd ControlNet-main/gradio
   python app.py

On your *local machine*, establish SSH port forwarding and access the interface:

- **Windows**: open *PowerShell*
- **macOS / Linux**: open *Terminal*

.. code-block:: bash

   ssh -CNg -L 6006:127.0.0.1:6006 root@connect.nmb1.seetacloud.com -p <PORT>

.. note::

   On the first connection, if asked *yes/no*, type ``yes``. Then enter the server password 
   (note: the password will not be displayed while typing or pasting).  
   If you see ``Permission denied``, it likely means the password entry failed. Please retry.

After connecting, open your browser at: ``http://localhost:6006`` to access the Gradio interface.

Interface Usage
---------------

1. **Upload an image**: Click the top-left corner to upload an input image.
2. **Enter the prompt**: In the *prompt* field, specify whether to generate a benign or malignant tumor image, for example:  
   ``a photo of a benign breast tumor`` or ``a photo of a malignant breast tumor``.
3. **Generate**: Click **Generate**. After a short wait, the right-hand side will display 
   the extracted **Laplacian edges** and the generated **SWE image**.

Option 2: Provided Script

.. code-block:: bash

    # Example: Run inference with a sample ultrasound (US) image
    python inference.py \
        --input_path ./examples/sample_us.png \
        --output_path ./results/sample_swe.png \
        --config ./configs/controlnet.yaml \
        --checkpoint ./checkpoints/controlnet.pth








