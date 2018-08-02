# Freesound General-Purpose Audio Tagging Challenge

Description of the approach to the [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging).

The code here is some stuff for running cross-validation stuff, feature processing and writing logs; the end result was a bit of a mess, so it's not going to be uploaded; see approach and description below.

# Approach

Result: top 11% (57th place), Private LB score: 0.8971, Public LB score: 0.9269.

I used two classifiers, LGBM and CatBoost, and used a geometric mean to find the top 3 categories: LGBM^0.75 + CatBoost^0.25.

## LightGBM settings

(`'sample_weights_manually_verified'` is the sample weight of the manually verified files in the train-set, other files have a sample weight of 1).
- `'n_estimators': 1400`
- `'num_leaves': 16`
- `'colsample_bytree': 0.2`
- `'learning_rate': 0.015`
- `'subsample': 0.74`
- `'subsample_freq': 5`
- `'reg_lambda': 0.023`
- `'min_child_samples': 30`
- `'class_weight': 'balanced'`
- `'sample_weights_manually_verified': 1.02`

## CatBoost settings:

- `'loss_function': 'MultiClass'`
- `'n_estimators': 1800`
- `'learning_rate': 0.012`
- `'rsm': 0.1`
- `'reg_lambda': 0.025`
- `'bagging_temperature': 0.75`
- `'depth': 6`
- `'random_seed': 4`
- `'sample_weights_manually_verified': 1.05`

## Feature processing

To reduce the number of features, features with an absolute correlation value higher than 0.9 were dropped (obviously, out of two highly correlated features, one was kept, and the other dropped), as well as features which had an importance less than 50 when running LGBM (this was done not on a full feature set, so in the end, some features with an importance lower than 50 crept in).

2145 total features, and 1540 features used after clean-up.

## Feature sets

### Raw waveform features

- Position of minimum and maximum of waveform
- Minimum and maximum of waveform
- Mean, standard deviation
- 10th, 25th, 50th, 75th, 90th percentiles
- Skew, kurtosis
- RMS of waveform
- Ratio of RMS to STD
- Ratio of Maximum to Minimum

### YAAFE features:

The following features were extracted using the YAAFE toolbox (see YAAFE docs for feature descriptions):

- MFCC (15 MFCC coefficients)
- Amplitue Modulation
- Energy
- Loudness (loudness in each Bark band)
- PerceptualSharpness
- PerceptualSpread
- SpectralFlux
- SpectralFlatness
- SpectralRolloff (using default block and step size, and using block size=512, step_size=256)
- SpectralVariation
- SpectralShapeStatistics
- OBSIR

Time derivatives (via `np.gradient`) were taken from the following features:
- SpectralVariation
- SpectralFlux
- SpectralFlatness
- Loudness
- MFCC
- SpectralShapeStatistics
- OBSIR

Second derivatives were taken for the MFCC coefficients.

For each of these features, the following functions (time-wise) were computed:

- Position of minimum and maximum
- Minimum and maximum
- Mean, standard deviation
- 10th, 25th, 50th, 75th, 90th percentiles
- Skew, kurtosis

### Autocorrelation features

Autocorrelation was computed for the wav files, as well as for Gammatone-filtered audio (using a set of 48 gammatone filters from 100 to 11025 Hz; using the [Gammatone Filterbank Toolkit](https://github.com/detly/gammatone)).

For these autocorrelation functions, the zero-crossing rate, position of first peak, and the normalized value of the first peak were computed.

The gammatone filter-based approach was proposed in [Valero X, Alías F. Narrow-band autocorrelation function features for the automatic recognition of acoustic environments. The Journal of the Acoustical Society of America. 2013 Jul;134(1):880-90.](https://www.ncbi.nlm.nih.gov/pubmed/23862894).

### Essentia features

The following features were extracted using the Essentia toolbox (see Essentia docs for feature descriptions):

- DerivativeSFX
- FlatnessSFX
- LogAttackTime
- StrongDecay
- TCToTotal
- HFC
- PitchSalience
- Inharmonicity
- Dissonance
- MaxMagFreq
- StrongPeak

The following features were also computed:
- Fundamental Frequency
- Confidence of fundamental frequency recognition
- Time derivative of fundamental frequency

For these three time-dependent features, the following functions (time-wise) were computed and used as features:
- Minimum and maximum
- Mean, standard deviation
- 10th, 25th, 50th, 75th, 90th percentiles
- Skew, kurtosis

### VGGish features:

[VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) is a pre-trained VGG-like CNN (trained on AudioSet) and was used to obtain 128-dimensional embeddings for each audio file.
See also [my modification](https://github.com/knstmrd/vggish-batch) which allows for batch processing and handling of short audiofiles.

In case of short audiofiles, they were repeated until their length exceeded 1 second. The embedding was averaged time-wise, and the resulting 128-dimensional vector used as a feature.

## Most important features

For a full list of feature importances obtained using the best-performing LGBM model, see `importances.md`.

The top 40 features are (and their importances):

- vggish 49, 1539
- vggish 40, 1541
- vggish 52, 1543
- autocorr_peak_position, 1566
- vggish 8, 1567
- vggish 27, 1571
- vggish 23, 1573
- logattack 0, 1584
- energy basic kurtosis, 1592
- vggish 58, 1593
- vggish 28, 1602
- vggish 14, 1630
- amplitudemod basic 5 max, 1634
- spectralflux basic perc10, 1650
- vggish 30, 1651
- maxmag, 1663
- vggish 61, 1666
- vggish 16, 1667
- autocorr_ZCR, 1671
- vggish 33, 1726
- vggish 66, 1749
- energy basic skew, 1765
- vggish 6, 1781
- vggish 7, 1788
- vggish 34, 1816
- vggish 3, 1877
- vggish 9, 1916
- vggish 10, 1925
- vggish 12, 1926
- vggish 45, 1947
- vggish 31, 1970
- vggish 69, 2033
- obsir basic 0 max, 2039
- logattack 1, 2088
- vggish 26, 2184
- wav kurtosis, 2210
- vggish 11, 2491
- derivative SFX 1, 2717
- salience, 2790
- vggish 0, 3028