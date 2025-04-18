# FingerECoG-BCI

Prediction of finger flexion from ECoG data - BCI Competition IV

## Models

By transforming the waveform timeseries ECoG data into spectrograms, we leverage on the new-found homophilic property of the spectrogram to train convolutional autoencoders (CAE) for prediction flexion patterns.

Our vanilla encoder-decoder model architecture is as follows. We trained the model on subject 1's data and got an overall Pearson correlation coefficient of **0.3501**, which is less than ideal.

![Vanilla CAE model](assets/VanillaEncoderDecoder.png)
_Vanilla CAE model with three conv-deconv layers, using batchnorm, dropout, and GeLU activation._

## Dataset

We use dataset 4 from BCI IV Competition. It contains ECoG signals recorded together with fingers movements.

More details about the dataset and the BCI competition can be found here: http://www.bbci.de/competition/iv/.

## Project Information

The project folder structure is organised as follows:

```text
.
├── data/
│   ├── finger_flex_cropped_val.npy
│   ├── finger_flex_cropped.npy
│   ├── sub1_comp.mat
│   ├── sub1_testlabels.mat
│   ├── sub2_comp.mat
│   ├── sub2_testlabels.mat
│   ├── sub3_comp.mat
│   ├── sub3_testlabels.mat
│   ├── X_spectrogram_cropped_val.npy
│   └── X_spectrogram_cropped.npy
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── encoder_decoder.ipynb
│   ├── eval.py
│   ├── model.py
│   ├── prepare_data.ipynb
│   ├── run_lstm.ipynb
│   └── train_sub1_lstm.ipynb
│   └── train_sub2_lstm.ipynb
│   └── train_sub3_lstm.ipynb
├── requirements.txt
└── sub1.pth
└── sub2.pth
└── sub3.pth
```

The `/data` folder contains all original and preprocessed data files used for training. This includes the cropped and up/down-sampled ECoG signals and the spectrogram conversion.

The `/src` folder contains Jupyter notebooks used to preprocess the data (`prepare_data.ipynb`), and the two models - the vanilla autoencoder (`encoder_decoder.ipynb`) and the CAE-LSTM hybrid (`train_subX_lstm.ipynb`, `run_lstm.ipynb`) - as described above. The remaining Python files are helper files used across the various Jupyter notebooks.

The best-performing models per subject have been saved into their respective `.pth` files, which can be loaded for testing.

### How to Use

1. Download the dataset and unzip the files into a `/data` folder.
2. Install the Python requirements using

```shell
pip install -r requirements.txt
```

3. Run the data preparation notebook, followed by the model of your choosing.

## Acknowledgements

This project is an undertaking of SUTD's 50.039 Theory and Practice of Deep Learning. We also reference the following research articles as cited below:

```
@article{bcicompiv2007,
  title={Decoding Two-Dimensional Movement Trajectories Using Electrocorticographic Signals in Humans},
  author={Schalk, G., Kubanek, J., Miller, K.J., Anderson, N.R., Leuthardt, E.C., Ojemann, J.G., Limbrick, D., Moran, D.W., Gerhardt, L.A., and Wolpaw, J.R.},
  journal={J Neural Eng, 4: 264-275},
  year={2007}
}

@article{fingerflex2022,
  title={FingerFlex: Inferring Finger Trajectories from ECoG signals},
  author={Lomtev, Vladislav and Kovalev, Alexander and Timchenko, Alexey},
  journal={arXiv preprint arXiv:2211.01960},
  year={2022}
}
```
