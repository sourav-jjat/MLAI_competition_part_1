# MLAI_competition_part_1
 This is a project for the MLAI competition 2020. The competition is promoted by ASTRO Lab
 
 ##Instructions for installing tensorflow in you windows system
 A concise method of installing tensorflow 2.x will be explained in here( without CUDA support). For a much detailed explanation please refer to [Tensorflow installation](https://readthedocs.org/projects/tensorflow-object-detection-api-tutorial/downloads/pdf/latest/).
 
 The software tools required are
 -OS Windows
 -Python 3.8+
 -Anaconda (optional)
 
 ## Tensorflow installation
 
 ### Installing Anaconda
 Anaconds package manager is really great so please install it through [here](https://www.anaconda.com/products/individual).
 
 ### Installing tensorflow
 1. Open a new terminal window of your choice ( command prompt, Anaconda powershell import).
 2. Type conda `create -n tensorflow pip python=3.8`
 3. In the same terminal type `conda activate tensorflow`
 After activation the terminal would like
 `(tensorflow) C:\Users\Astrolab>`
 4. `pip install --ignore-installed --upgrade tensorflow==2.2.0`
 
 ### Verifying the installation
 In a new terminal run
 `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))`
 you might see a lot of output but in the end you would see 
 `tf.Tensor(some number, shape=(), dtype=float32)`
 
 ## TensorFlow Object Detection API installation
 
 ### Download Tensoflor Model Garden
 1. Create a new folder and name it Tensorflow `C:\Users\Astro Lab\Documents\TensorFlow`
 2. Open Terminal `cd` into the `Tensorflow` directory
 3. Use git to clone the [repository] or download it as a zip folder and extract it. Rename the folder models-master to models.
 
 
 ## Protobuf Installation
 1. Access the site for installing [protc](https://github.com/protocolbuffers/protobuf/releases)
 2. Download protoc-X.XX.X-wind64.zip
 3.Extract content to a directory of choice or C:\Program Files\Google Protobuf
 4.Add the above path to you Path environment varaible
 5. Open a new terminal and `cd` into `TensorFlow/models/research/` and type `protoc object_detection/protos/*.proto --python_out=.`
 


 




 
 
 
 
 
 
