# NonAdjacencyLoss

Image segmentation based on convolutional neural networks is proving to be a powerful and efficient solution for medical applications. However, the lack of annotated data, presence of artifacts and variability in appearance can still result in inconsistencies during the inference. We choose to take advantage of the invariant nature of anatomical structures, by enforcing a semantic constraint to improve the robustness of the segmentation. The proposed solution is applied on a brain structures segmentation task, where the output of the network is constrained to satisfy a known adjacency graph of the brain regions. This criteria is introduced during the training through an original penalization loss named NonAdjLoss. With the help of a new metric, we show that the proposed approach significantly reduces abnormalities produced during the segmentation. Additionally, we demonstrate that our framework can be used in a semi-supervised way, opening a path to better generalization to unseen data.

The NonAdjacency loss can be applied to any pre-trained models, you will first need to extract the adjacency prior from an annotated training set and finally re-train your model to enforce the constraint on the model's output. You can also apply it on new images without ground truth !

### Example MICCAI 2012 multi-atlas segmentation challenge

An example implementation for this brain segmentation challenge, with adjacency
penalization by the NonAdjLoss are available in our in-house pytorch library for medical imaging. [HERE](https://github.com/trypag/torchmed/tree/master/examples/02_segmentation_NonAdjLoss)

### Adjacency graph extraction

Use the following script to extract the adjacency map from the dataset :

```bash
python extract_adjacency_matrix.py data output nb_labels
```

with `data` the input dataset, `output` the output directory and `nb_labels` the number
of labels. The `data` directory should be organized with one folder for each exam and
the segmentation map named `prepro_seg_mni.nii.gz`.

Here is how the data should look like :

```bash
[ganaye@iv-ms-593 train]$ ll .
total 516
drwxrwxr-x 2 ganaye creatis   4096 Jan 23 11:09  1000
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1001
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1002
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1006
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1007
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1008
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1009
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1010
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1011
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018  1012
[ganaye@iv-ms-593 train]$ ll 1000/
total 41864
-rw-rw-r-- 1 ganaye creatis     2670 Jan 15  2018 graph.png
-rw-rw-r-- 1 ganaye creatis   455625 Jan 15  2018 graph.png.csv
-rw-rw-r-- 1 ganaye creatis 16340661 Jan 15  2018 im_mni_bc.nii.gz
-rw-rw-r-- 1 ganaye creatis  8589615 Jan 15  2018 im_mni.nii.gz
-rw-rw-r-- 1 ganaye creatis      125 Jan 15  2018 mni_aff_transf.c3dmat
-rw-rw-r-- 1 ganaye creatis      190 Jan 15  2018 mni_aff_transf.mat
-rw-rw-r-- 1 ganaye creatis 16761108 Jan 15  2018 prepro_im_mni_bc.nii.gz
-rw-rw-r-- 1 ganaye creatis   344875 Jan 15  2018 prepro_seg_mni.nii.gz
-rw-rw-r-- 1 ganaye creatis   344193 Jan 15  2018 seg_mni.nii.gz

```

### NonAdjacency Loss

Example use case :

```bash
>>> import torch
>>> import numpy
# requires torch v1.0 for torch.einsum
>>> torch.__version__
'1.0.0a0+db5d313'
# import adjacency estimator
>>> from loss import AdjacencyEstimator

# load adjacency matrix
>>> graph_gt = torch.from_numpy(numpy.loadtxt('graph.png.csv', delimiter=';'))
# binarize the matrix and negate, to obtain forbidden adjacency matrix
>>> graph_gt = 1 - (graph_gt > 0).float()
>>> nb_conn_ab_max = (graph_gt > 0).sum()
# number of labels (anatomical structures)
>>> nb_classes = 135
>>> adj_layer = AdjacencyEstimator(nb_classes)

# net should be your architecture, outputing a batch of 2D probability maps
>>> output = net(input)
# output should be a tensor of size batch x channels x h x w
# output is a tensor of probability maps (each vector sums to 1) (output of softmax)
>>> adj_loss = adj_layer(output)
# compute loss only for forbidden adjacencies
>>> non_adj_loss = gt_graph * adj_loss
# normalize by the number of forbidden connections
>>> non_adj_loss = non_adj_loss.sum() / nb_conn_ab_max
# weight the loss with lambda
>>> non_adj_loss = non_adj_loss * lambda_coef
>>> loss = dice + cross_entropy + non_adj_loss

>>> loss.backward()
```
