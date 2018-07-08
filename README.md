# Zero shot Raven's Progressive Matrix solver (Work in progress)

Solving [Raven's Progressive Matrix](https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices) using an autoencoder configuration without any prior training.

# Results

## Obtained
![](./imgs/1000_CNN_CNN.png)

## Expected
![](./data/sample0/a3.png)
![](./data/sample0/b3.png)
![](./data/sample0/o8.png)

## PSNR
![](./imgs/PSNR.png)

The diagram shows the PSNR between the available options and the predicted output. Each color represents an epoch and each of the eight bars of same color represent the eight available options.

 The last option is the correct answer and has the highest PSNR as expected.