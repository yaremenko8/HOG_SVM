# HOG_SVM
This is just a simple implementation of a classifier that uses [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) descriptors. Just wanted to code this myself to get the hang of it. 
<br>Labeled training samples that were used for testing can be found [here](https://docviewer.yandex.ru/view/108330691/?*=BJfvbQ6aSc2TTb%2BDxYAO1MMM7vR7InVybCI6InlhLWRpc2stcHVibGljOi8veU9PVGNFeHY2cVJSb2dqSEUySHpUdXZkYTdCWXZrUFBrQVFCWFFCRTk0dz06L2RhdGEvdHJhZmZpYy1zaWducy10cmFpbi56aXAiLCJ0aXRsZSI6InRyYWZmaWMtc2lnbnMtdHJhaW4uemlwIiwidWlkIjoiMTA4MzMwNjkxIiwieXUiOiI5ODU3MzYzNjExNDg4ODAyNTcxIiwibm9pZnJhbWUiOmZhbHNlLCJ0cyI6MTQ5MjYxNjQxOTI4OH0%3D).
<br><br><br>
## hog.py
Contains the descriptor extraction function ```extract_hog(img)```, where ```img``` is either a path string or an image data array.
<br><br>
## svm_train.py
Contains the training function ```train(gamma_, C_, pool)```, where ```gamma_``` and ```C_``` are the according SVC parameters and ```pool``` is a multiprocessing pool object. The function extracts descriptors of the given dataset: ```samples_from_class``` and ```tests_from_class``` samples per existing label for training and testing accordingly. Provided image files should be stored in the directory specified by ```path``` variable. The directory should also include a comma-seperated values file ```gt.csv``` of "file, label" rows sorted in ascending order by label. After extracting descriptors and training an SVC, the function computes and displays the accuracy score achieved by the resulting machine over the testing set. The function then provides the means to save the machine in a separate file.
<br><br>
## fit_and_classify.py
Contains the ```fit_and_classify(machine_file, samples)``` function, where ```machine_file``` is the path to the SVC-machine file used for evaluation and ```samples``` is the data set to be evaluated.
<br><br>
## gamma_c_grid.txt
Contains the output of a sequence that evaluated ```train(gamma_, C_, pool)``` over a logarithmic grid of C and gamma parameters with ```samples_from_class = 195``` and  ```tests_from_class = 25```. 
