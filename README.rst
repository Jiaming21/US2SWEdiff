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

Step 3: Run Inference
======================

After completing the environment setup and cloning the repository (see Step 1 and Step 2), 
you can perform inference using either the **Gradio** graphical interface or command line.

.. contents::
   :local:
   :depth: 2

Option 1: Using the Gradio Interface
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

    .. note::

       - On the first connection, if prompted with *yes/no*, type ``yes``.  
       - Enter the server password (it will not be displayed while typing or pasting — this is normal).  
       - If you see ``Permission denied``, the password was likely incorrect. Please retry.

    After connecting, open your browser at: ``http://localhost:6006`` to access the Gradio interface.

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~

    If you prefer to run everything directly on your **local computer**:

    On your **local terminal** (PowerShell for Windows, or Terminal for macOS/Linux):

    .. code-block:: bash

       cd ControlNet-main/gradio
       python app.py

    Once the Gradio server has started, the terminal will display something like:

    .. code-block:: text

       Running on local URL:  http://127.0.0.1:7860/

    Now open your browser and go to the displayed URL (commonly ``http://127.0.0.1:7860`` or ``http://localhost:7860``) to access the interface.

**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

1. **Upload an image**: Click the top-left window to upload your input image.
2. **Enter the prompt**: In the *prompt* field, type your description, e.g.:  
   ``a photo of a benign breast tumor`` or ``a photo of a malignant breast tumor``.
3. **Generate**: Click **Generate**. After a short wait, the right-hand panel will display 
   the extracted **Laplacian edges** and the generated **SWE image**.



Option 2: Provided Scripts
------------------------------------

.. code-block:: bash

    # Example: Run inference with a sample ultrasound (US) image
    python inference.py \
        --input_path ./examples/sample_us.png \
        --output_path ./results/sample_swe.png \
        --config ./configs/controlnet.yaml \
        --checkpoint ./checkpoints/controlnet.pth








