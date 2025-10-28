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
    * `Step 4: Run Inference <#step-4-run-inference>`_
    
	* `Option 1: Using Gradio Interface <#option-1-using-gradio-interface>`_

        	* `Run on Remote Server <#run-on-remote-server>`_
		* `Run on Local Computer <#run-on-local-computer>`_
		* `Gradio Interface Usage Instructions <#gradio-interface-usage-instructions>`_

        * `Option 2: Using Provided Scripts <#option-2-using-provided-scripts>`_

    		* `Step 1–3: Prepare Project Environment <#step-13-prepare-project-environment>`_
    		* `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file>`_
    		* `Step 5: Build the Inference Dataset <#step-5-build-the-inference-dataset>`_
    		* `Step 6: Load the ControlNet Model <#step-6-load-the-controlnet-model>`_

* `Train <#train>`_

    * `Step 1–2: Prepare Project Environment <#step-12-prepare-project-environment>`_
    * `Step 3: Prepare the Dataset <#step-3-prepare-the-dataset-train>`_
    * `Step 4: Create the "metadata.json" File <#step-4-create-the-metadatajson-file-train>`_
    * `Step 5: Build the Training Dataset <#step-5-build-the-training-dataset>`_
    * `Step 6: Create Complete Model Weights <#step-6-create-complete-model-weights>`_
    * `Step 7: Load and Train the Model <#step-7-load-and-train-the-model>`_

* `Advanced Options for Training <#advanced-options-for-training>`_





.. raw:: html

   <hr>





.. _inference:

Inference
=============

.. _step-1-prepare-conda-environment:

Step 1: Prepare Conda Environment
=================================
First install `Anaconda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. Then, create environment and install packages and dependencies using following command (here CUDA 11.3):

.. code-block:: bash

    # Create a new environment named "controlnet" with Python 3.10
    conda create -n controlnet python=3.10

    # Activate the environment
    conda activate controlnet

    # Install dependencies from controlnet.yaml (environment reproduction)
    conda env update -n controlnet -f controlnet.yaml

This will create a conda environment named ``controlnet`` with packages and dependencies installed.






.. _step-2-pull-from-github-repository:

Step 2: Pull from GitHub Repository
===================================

Clone the US2SWEdiff repository from GitHub:

.. code-block:: bash

    git clone https://github.com/Jiaming21/US2SWEdiff.git
    cd US2SWEdiff

.. raw:: html

   <details>
   <summary><strong>Model Files</strong> (click to expand)</summary>

   <p>
     The large model files used in this project (<code>stable-diffusion-v1-5</code> and
     <code>clip-vit-large-patch14</code>) are stored separately on the 🤗 Hugging Face Hub
     for size and licensing reasons.
   </p>

   <p>
     For more information about these models and their usage conditions, please refer to:
     <code>models/model_files_notice.txt</code>
   </p>

   <p>Or visit the model pages directly:</p>
   <ul>
     <li>Stable Diffusion v1.5: <a href="https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5">https://huggingface.co/Jiaming2143183/stable-diffusion-v1-5</a></li>
     <li>CLIP ViT-L/14: <a href="https://huggingface.co/Jiaming2143183/clip-vit-large-patch14">https://huggingface.co/Jiaming2143183/clip-vit-large-patch14</a></li>
   </ul>

   <hr>

   <p><strong>After downloading</strong>, drag the <code>stable-diffusion-v1-5</code> and
   <code>clip-vit-large-patch14</code> folders into the <code>models/</code> directory.</p>

   <h4>Verify script paths and weights</h4>

   <p>You should also check the following script point to the correct model weights.</p>

   <p><code>[your_path_to_ControlNet-main_folder]/ldm/modules/encoders/modules.py</code></p>

   <p>
     In class <code>FrozenCLIPEmbedder</code> in the <code>__init__</code> function,
     change the version to
     <code>[your_path_to_ControlNet-main_folder]/models/clip-vit-large-patch14</code>.
   </p>

   <p>
     As for the <code>stable-diffusion-v1-5</code> folder, the
     <code>v1-5-pruned.ckpt</code> file inside will be used to create complete weights with
     <code>[your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py</code>
     in the Training section's <strong>Step 6: Create Complete Model Weights</strong>.
   </p>

   </details>







.. _step-3-prepare-the-dataset:

Step 3: Prepare the Dataset
===================================

*(This step is only required if you wish to apply the model infer your own dataset. For this project, all data are already well organized when you clone the repository.)*

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

.. raw:: html

   <details>
   <summary><strong>Advanced Options (click to expand)</strong></summary>

   <ul>
     <li><strong>Images</strong> — the number of images to generate.</li>
     <li><strong>Laplacian ksize (odd)</strong> — the kernel size used by the Laplacian edge detector (odd integers only: 1, 3, 5, 7, …).
       <br>Smaller values give finer, sharper edges; larger values give thicker, smoother edges (with more noise suppression).
     </li>
   </ul>

   </details>



.. raw:: html

   <details>
   <summary><strong>Adjustment if not "PNG, RGB, 8bit" combination (Click to open)</strong></summary>

   <p>
   If your images are <b>not in PNG, RGB, 8-bit format</b>, you need to modify the following code in  
   <code>[your_path_to_ControlNet-main_folder]/tutorial_dataset.py</code>.
   </p>

   <p><b>1. If the image format is different:</b><br>
   Change the image loading mode by editing the following two lines:</p>

   <pre><code>source = Image.open(source_filename).convert('RGB')
target = Image.open(target_filename).convert('RGB')
   </code></pre>

   <p><b>2. If the bit depth is different:</b><br>
   Modify the Mask-Image Pair processing section as follows:</p>

   <pre><code>source = np.array(source).astype(np.uint8)
target = np.array(target).astype(np.uint8)

source = source.astype(np.float32) / 255.0
target = target.astype(np.float32) / 127.5 - 1.0
   </code></pre>

   <p><b>3. If the image size is different:</b><br>
   No problem — the <code>transform</code> function will automatically resize images to <b>256×256</b>.</p>

   <p><b>4. If your images are saved in another format:</b><br>
   To ensure the training images are saved in the correct format, modify the following two functions in  
   <code>[your_path_to_ControlNet-main_folder]/cldm/logger.py</code>:</p>

   <ul>
     <li><code>log_local</code></li>
     <li><code>log_img</code></li>
   </ul>

   <p>This ensures that all training and logged images are saved in your specified image format.</p>

   </details>

<!-- 👇 关键：这两个空行非常重要！一定要保留！ -->



.. _step-4-run-inference:

Step 4: Run Inference
===================================

After completing the environment setup, cloning the repository, and preparing the dataset (see Step 1-3 above), you can perform inference using either the **Gradio** graphical interface or command line.





.. _option-1-using-gradio-interface:

Option 1: Using Gradio Interface
--------------------------------

You can run the Gradio interface in **two ways**:

1. On a **remote server** with SSH port forwarding.
2. Directly on your **local computer**.






.. _run-on-remote-server:

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






.. _run-on-local-computer:

**Run on Local Computer**
~~~~~~~~~~~~~~~~~~~~

    On your *local terminal* (PowerShell for Windows, or Terminal for macOS/Linux):

    .. code-block:: bash

       cd ControlNet-main/gradio
       python app.py

    Once the Gradio server has started, the terminal will display something like:

    .. code-block:: text

       Running on local URL:  http://127.0.0.1:7860/

    Now open your browser and go to the displayed URL (commonly ``http://127.0.0.1:7860`` or ``http://localhost:7860``) to access the interface.



**Gradio Interface Usage Instructions**
~~~~~~~~~~~~~~~~~~~~

.. image:: https://raw.githubusercontent.com/Jiaming21/US2SWEdiff/main/github_img/gradio.png
   :width: 1000

In the Gradio interface above, follow the steps below to run the inference:

1. **Upload an image**: Click the top-left window to upload your input image.
2. **Enter the prompt**: In the *prompt* field, type your description, e.g.:  
   ``a photo of a benign breast tumor`` or ``a photo of a malignant breast tumor``.
3. **Generate**: Click the **Generate** button. After a short wait, the right-hand panel will display 
   the extracted **Laplacian edge** and the generated **SWE images**.

.. raw:: html

   <details>
   <summary><strong>Advanced Options (click to expand)</strong></summary>

   <ul>
     <li><strong>Images</strong> — the number of images to generate.</li>
     <li><strong>Laplacian ksize (odd)</strong> — the kernel size used by the Laplacian edge detector (odd integers only: 1, 3, 5, 7, …).
       <br>Smaller values give finer, sharper edges; larger values give thicker, smoother edges (with more noise suppression).
     </li>
   </ul>

   </details>




.. raw:: html

   <hr>




.. _option-2-using-provided-scripts:

Option 2: Using Provided Scripts
--------------------------------

In the following example, we demonstrate the best-performing model proposed in our paper — the *"Laplacian edge map → SWE image"* approach — applied to the public **BUSI** dataset for inference.

.. _step-13-prepare-project-environment:

Step 1–3: Prepare Project Environment
======================================

Repeat **Step 1-3** from the *Inference* section to set up the environment，clone the repository and prepare the dataset for inference.


.. _step-4-create-the-metadatajson-file:

Step 4: Create the ``metadata.json`` File
=========================================

First, navigate to the following directory:

.. code-block:: bash

   cd [your_path_to_ControlNet-main_folder]/data/tools/

Under this directory, there is a script named ``data.py``.  
Modify this file to ensure that the paths are correctly specified.

The following lines should be checked and updated accordingly:

.. code-block:: python

   imagepath = "../infer/BUSI/*"  # Since we are performing inference, this can point to any image folder
   condpath = "../infer/laplacian/"  # Path to your condition images (here we use Laplacian edge maps)

   root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/"  # Path to your dataset root directory

   with open("../infer/metadata.json", 'w') as f:  # This will be your newly created metadata JSON file

After verifying all paths, run the following command to generate the metadata file:

.. code-block:: bash

   python data.py

Once completed, the JSON file will be created under the designated ``../infer/metadata.json`` folder.






.. _step-5-build-the-inference-dataset:

Step 5: Build the Inference Dataset
===================================

Build the dataset for inference using the previously generated ``metadata.json`` file.

1. Open the following script:

   .. code-block:: text

      [your_path_to_ControlNet-main_folder]/tutorial_dataset.py

2. Locate the ``MyDataset`` class and modify the ``root`` variable as shown below:

   .. code-block:: python

      root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/infer/BUSI/metadata.json"

This ensures that the dataset is correctly built based on the metadata file created in **Step 4**.







.. _step-6-load-the-controlnet-model:

Step 6: Load the ControlNet Model
=================================

Load the ControlNet model (refer to ``cldm/cldm.py``) with your previously trained weights.

Your model checkpoints are stored under the following directory:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/lightning_logs/

For example, if you wish to use the following trained checkpoint:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt

You need to open the following script:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/tutorial_inference.py

Then, modify the following variables within the script to match your paths:

.. code-block:: python

   CKPT_PATH = "[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/epoch=129-step=6110.ckpt"
   RESULT_DIR = "[your_path_to_ControlNet-main_folder]/generated_results/"

After saving the modifications, run the script:

.. code-block:: bash

   python [your_path_to_ControlNet-main_folder]/tutorial_inference.py

The generated inference results will be saved in the following directory:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/generated_results/version_0/




.. raw:: html

   <hr>





.. raw:: html

   <a id="train"></a>
   <details>
   <summary><h2><strong>Train (click to expand)</strong></h2></summary>


In the following example, we demonstrate the training of the best-performing model proposed in our paper,  
which uses the **Laplacian edge map** as the conditioning input to generate the corresponding **SWE image**.


.. _step-12-prepare-project-environment:

Step 1–2: Prepare Project Environment
=======================================

Repeat Step 1–2 from *Inference* to prepare conda environment and pull from GitHub repository.


.. _step-3-prepare-the-dataset-train:

Step 3: Prepare the Dataset
=======================================

*(This step is only required if you wish to train the model on your own dataset.  
For this project, all data are already well organized when you clone the repository.)*

The dataset directory structure should look like this:

.. code-block:: text

    Breast-img/
    └── Train/
        ├── us/
        ├── canny/
        ├── laplacian/ （used condition images folder for this example）
        └── swe/ （used target images folder for this example）

Each subfolder under ``Train/`` should contain your corresponding images in standard formats (e.g., ``.png``, ``.jpg``, or ``.tif``).


.. _step-4-create-the-metadatajson-file-train:

Step 4: Create the ``metadata.json`` File
=======================================

Navigate to the following directory:

.. code-block:: bash

   cd [your_path_to_ControlNet-main_folder]/data/tools/

Within this directory, you will find the script ``data.py``.  
Modify this file to ensure that all paths are correctly set to your dataset locations.

The key sections of the code that need to be updated are as follows:

.. code-block:: python

   imagepath = "../train/swe/"        # Make sure this points to the target images (i.e., SWE images) folder
   condpath = "../train/laplacian/"   # Make sure this points to your condition images (Laplacian edge maps)

   root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/"  # Ensure this points to the correct data path

   with open("../train/metadata.json", 'w') as f:  # This will be your newly created metadata file

After confirming that all paths are correct, run the following command:

.. code-block:: bash

   python data.py

This will create the ``metadata.json`` file under the specified directory:

.. code-block:: text

   ../train/metadata.json


.. _step-5-build-the-training-dataset:

Step 5: Build the Training Dataset
==================================

Build the dataset for training using the previously created ``metadata.json`` file.

Open the following script:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/tutorial_dataset.py

Within the script, locate the definition of the ``MyDataset`` class and modify the ``root`` variable as follows:

.. code-block:: python

   root = "[your_path_to_ControlNet-main_folder]/data/BreastCA-img/train/metadata.json"

This ensures that your dataset loader correctly reads the training data defined in the ``metadata.json`` file.


.. _step-6-create-complete-model-weights:

Step 6: Create Complete Model Weights
=====================================

In this step, you will create the complete model weights (i.e., ``controlnet.ckpt = SD + ControlNet``)  
for the ControlNet model (refer to ``cldm/cldm.py``).

Here we use ``stable-diffusion-v1-5/v1-5-pruned.ckpt`` as the pretrained Stable Diffusion weights  
to generate the combined ControlNet checkpoint.

Run the following command:

.. code-block:: bash

   python [your_path_to_ControlNet-main_folder]/ControlNet-main/tool_add_control.py \
       [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/v1-5-pruned.ckpt \   # SD-only weights
       [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/controlnet.ckpt       # Output combined SD + ControlNet weights

After running the script, a file named ``controlnet.ckpt`` will be created under:

.. code-block:: text

   [your_path_to_ControlNet-main_folder]/ControlNet-main/models/stable-diffusion-v1-5/

This file represents the **complete pretrained weights** required for initializing ControlNet training.


.. _step-7-load-and-train-the-model:

Step 7: Load and Train the Model
================================

To begin training, ensure that you are using the correct **complete pretrained weights** generated in the previous step.

Set the following path inside your training script:

.. code-block:: python

   resume_path = "[your_path_to_ControlNet-main_folder]/models/stable-diffusion-v1-5/controlnet.ckpt"  # Ensure this uses the correct complete pretrained weights

Then, run the following command to start training:

.. code-block:: bash

   python [your_path_to_ControlNet-main_folder]/ControlNet-main/tutorial_train.py


.. raw:: html

   <details>
   <summary><strong>Training Outputs (click to expand)</strong></summary>

   <p>After successful execution, the training process will generate the following outputs:</p>

   <ol>
     <li><strong>Model Checkpoints (Full Architecture)</strong><br>
         Stored under:<br>
         <code>[your_path_to_ControlNet-main_folder]/lightning_logs/version_1/checkpoints/</code><br>
         Example:<br>
         <code>epoch=129-step=6110.ckpt</code>
     </li>
     <br>
     <li><strong>Training Image Logs</strong><br>
         Located at:<br>
         <code>/root/autodl-tmp/ControlNet-main/image_log/train/</code><br>
         <p>This folder includes four visualization types:</p>
         <ul>
           <li><strong>Conditioning</strong> — Prompt (e.g., “a photo of a benign/malignant breast tumor”)</li>
           <li><strong>Control</strong> — Laplacian edge map</li>
           <li><strong>Reconstruction</strong> — True SWE images</li>
           <li><strong>Samples</strong> — Synthesized SWE images</li>
         </ul>
     </li>
   </ol>
   </details>

.. raw:: html

   </details>




.. raw:: html

   <a id="advanced-options-for-training"></a>
   <details>
   <summary><h2><strong>Advanced Options for Training (click to expand)</strong></h2></summary>

   <ul>
     <li><strong>Improved Hint Input Block</strong></li>
     <li><strong>Unlocked Decoder</strong></li>
     <li><strong>Classifier-free Guidance</strong></li>
   </ul>

   </details>
