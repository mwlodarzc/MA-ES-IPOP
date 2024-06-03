# MA-ES-IPOP
MA-ES extended with IPOP heurystic

## Run
To run and recreate results please follow instructions on installing the [coco bbob examples](https://github.com/numbbo/coco).
Instructions work for Python<=3.7

Than run CMAES optimization by:
```
python run_optimization.py --algorithm CMAES --suite bbob --instance 1 --dimensions 2 --seed 42 --result_folder my_experiment --run_cocopp
```

and MAES by calling:
```
python run_optimization.py --algorithm MAES --suite bbob --instance 1 --dimensions 2 --seed 42 --result_folder my_experiment --run_cocopp 

```

Script ```post_process.py``` creates charts out of results.
To run it call:
```
python post_process.py RESULTS_DIRECTORY
```
The same is done when flag ```--run_cocopp``` is set in ```run_optimization.py``` call.