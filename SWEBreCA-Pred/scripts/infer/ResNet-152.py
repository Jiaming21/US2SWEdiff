import runpy
import sys


if __name__ == "__main__":
    # Reuse the maintained inference implementation in ResNet-50.py,
    # which has been switched to ResNet-152 defaults.
    sys.argv[0] = __file__
    runpy.run_path(
        "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/SWEBreCA-Pred/scripts/infer/ResNet-50.py",
        run_name="__main__",
    )
