# FingerECoG-BCI

Prediction of finger flexion from ECoG data - BCI Competition IV

## Dataset

This is BCI IV Competition dataset 4. It contains ECoG signals recorded together with fingers movements.

More details about the BCI IV Competition: http://www.bbci.de/competition/iv/

'subN_comp.mat' files contain training ('train_data') and test ('test_data') ECoG signals and corresponding training fingers movements ('train_dg').

'subN_testlabels.mat` files contain test fingers movements ('test_dg') corresponding to the test dataset from 'subN_comp.mat' file.

The data are drawn from the 'fingerflex' data of Kai J. Miller's ECoG library, which can be downloaded at https://searchworks.stanford.edu/view/zk881ps0522.

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
