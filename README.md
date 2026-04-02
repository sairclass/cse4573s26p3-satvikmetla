[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6kfifRyc)
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Fqsa7e3B)
# S26-CSE473-573-A-P3

S26 CSE4/573 Project 3

**CSE 473/573 Face Detection and Recognition Project.**

The following commands should be executed in the root folder of the project.

Please first update below and push your change to this repo.

For the environment, you can use the project 2 setting and install face_recognition. Or use your own, there are no hard requirement about pytorch version.

- Your name   
- Your ubit name

**task 1 validation set**

```bash
# Face detection on validation data
python task1.py --input_path validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth validation_folder/ground-truth.json
```

**task 1 test set running**

```bash
# Face detection on test data
python task1.py --input_path test_folder/images --output ./result_task1.json
```

**task 2 running**

```bash
python task2.py --input_path faceCluster_5 --num_cluster 5
```


**Pack your submission**
Note that when packing your submission, the script would run your code before packing.

```bash
sh pack_submission.sh <YourUBITName>
```

* Note: In the commands, use `python3` if your environment has python named as `python3` instead `python`.

Change **`<YourUBITName>`** with your UBIT name.
The resulting zip file should be named **"submission\_`<YourUBITName>`.zip"**, and it should contain the following files: **"result\_task1.json"**, **"result\_task1\_val.json"**,**"result\_task2.json"**,**"face.py"**,**"viz_task1.png"**,**"viz_task2.png"**. If not, there is something wrong with your code/filename, please go back and check.

You should only submit the zip file.
