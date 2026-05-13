# Environment Setup

This project uses a local Miniforge prefix environment at `E:\tianchi_car\.conda`.

## Activate

In PowerShell:

```powershell
& "C:\Users\makab\Miniforge3\Scripts\activate"
conda activate E:\tianchi_car\.conda
```

Or run the interpreter directly:

```powershell
E:\tianchi_car\.conda\python.exe
```

## Installed Core Packages

- `python 3.11`
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `catboost`
- `xgboost`
- `matplotlib`
- `seaborn`
- `pyarrow`
- `jupyter`

## Recreate

If you want to rebuild the same environment:

```powershell
& "C:\Users\makab\Miniforge3\Scripts\conda.exe" create --yes --prefix "E:\tianchi_car\.conda" python=3.11 pandas numpy scikit-learn lightgbm catboost xgboost jupyter matplotlib seaborn pyarrow
```

Or from `environment.yml`:

```powershell
& "C:\Users\makab\Miniforge3\Scripts\conda.exe" env create --prefix "E:\tianchi_car\.conda" --file environment.yml
```

## Optional: Register Jupyter Kernel

```powershell
& "E:\tianchi_car\.conda\python.exe" -m ipykernel install --user --name tianchi-car --display-name "Python (tianchi-car)"
```
