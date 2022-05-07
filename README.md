# Subtype-Former
Subtype-Former is a deep learning cancer subtyping method based on multi-omics input. The input of Subtype-Former includes copy number, mRNA, miRNA, and DNA methylation, and the output is the corresponding cancer subtype for each sample. Meanwhile, Subtype-Former can automatically determine the number of subtypes when the subtype number is not specified. Subtype-Former runs fast and can work for pan-cancer tasks with a large amount of data.
```{r}
# We can use the following command to finish the subtyping process (We take BRCA as an example): 
python SubtypeFormer.py -m SubtypeFormer -c BRCA   
# the Clustering output file are stored in ./results/subtype_1/BRCA.SubtypeFormer  
```

Subtype-Former's Consensus clustering module is used as follows:  
```{r}
python SubtypeFormer.py -m cc -c BRCA
# record the corresponding class label for each sample and the output file is ./results/cc/BRCA.cc
```

Subtype-Former also supports the calculation of NMI and ARI: 
```{r}
python SubtypeFormer.py -m SubtypeFormer -c BRCA -d dataset_2
python SubtypeFormer.py -m nmiari -c BRCA
# record the corresponding class label for each sample and the output file is ./results/nmiari/BRCA.SubtypeFormer
```

Finally, we give a method to calculate all the results of Subtype-Former easily: 
```{r}
python run_all.py
# record the corresponding class label for each sample and the output file is ./results
```

Subtype-Former is based on the Python program language. The version of python is 3.8. We also use some open-source libraries: scikit-learn 0.23.2, pandas 1.1.3, numpy 1.19.2, pytorch 1.8.0. Subtype has been working on the Windows platform correctly. We used the NVIDIA GTX 2060 (8G) and Intel 10400F for the training.
