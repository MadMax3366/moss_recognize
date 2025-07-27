# cv_pic_2_coverage

Estimate moss coverage in images using computer vision


Overview
	1.	Preprocess raw images and extract quadrat (sample frame) regions.
	2.	Use CV methods to identify moss within the quadrat and calculate coverage rate.
	3.	Output statistical results: (image_name, coverage).

Environment

Designed to run on Linux/macOS.
Requires Python 3.12.

1. Create a virtual environment:
   python -m venv test_env
2. Activate the virtual environment:
   source ./test_env/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Prepare a folder containing your quadrat (sample plot) images.
   python ./run_light.py

