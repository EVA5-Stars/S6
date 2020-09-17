# **EVA5-Assignment-6**

## Group Members:
- Gaurav Makkar
- Mohit Bhandari
- Suman Debnath
- Sriram Iyengar


The objective of this project is to take best 4th code from S5, and run below versions for 25 epochs and report findings:

with L1 + BN
with L2 + BN
with L1 and L2 with BN
with GBN
with L1 and L2 with GBN
Draw ONE graph to show the validation accuracy curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.

Draw ONE graph to show the loss change curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.

Find any 25 misclassified images (combined into single image) for "with GBN" model. Use the saved model from the above jobs and show the actual and predicted class names.

The code is implemented in a highly modularized way and a single function was used to iterate through these conditions

Following are the accuracy observed in the last Epoch:

with L1 + BN => 99.29%
with L2 + BN => 99.35%
with L1 and L2 with BN => 99.47%
with GBN => 99.31%
with L1 and L2 with GBN => 99.35%