from importlib.metadata import version, PackageNotFoundError
for p in ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "hyperopt",
          "mlflow", "streamlit", "ipykernel", "notebook", "jupyterlab"]:
    try:
        print(f"{p}=={version(p)}")
    except PackageNotFoundError:
        pass          # skip tools you do not use