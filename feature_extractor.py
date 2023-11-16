import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import argparse
from pathlib import Path

# Overdamped Harmonic Oscillator curve fitting
def overdamped_harm_osc(x, A, B, w0):
    return (A + B * x) * np.exp(-w0 * x)

# Polynomial curve fitting
def polynomial5(x, a, b, c, d, e, f):
    return a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power(x, 4) + f * np.power(x, 5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--aorta', type=Path, required=True, dest='aorta_data')
    parser.add_argument('-b', '--brach', type=Path, required=True, dest='brach_data')
    parser.add_argument('-t', '--testing', action="store_true", required=False, dest="testing")
    args = parser.parse_args()

    #Load the files
    aorta_data = pd.read_csv(args.aorta_data)
    brach_data = pd.read_csv(args.brach_data)

    pidstr = "Unnamed: 0"

    #Get the waveform data by pid
    waveform_data = {}
    for pid in aorta_data[pidstr].values:
        waveform_data[pid] = {}
        aorta = aorta_data[(aorta_data[pidstr] == pid)]
        brach = brach_data[(brach_data[pidstr] == pid)]
        if args.testing:
            waveform_data[pid]['aorta'] = np.array(aorta).flatten()[1:]
            waveform_data[pid]['brach'] = np.array(brach).flatten()[1:]
        else:
            waveform_data[pid]['aorta'] = np.array(aorta).flatten()[1:-1]
            waveform_data[pid]['brach'] = np.array(brach).flatten()[1:-1]
        waveform_data[pid]['time_aorta'] = [(i, x) for i, x in enumerate(waveform_data[pid]['aorta']) if not np.isnan(x)]
        waveform_data[pid]['time_brach'] = [(i, x) for i, x in enumerate(waveform_data[pid]['brach']) if not np.isnan(x)]
        waveform_data[pid]['aorta_mean10'] = (pd.Series(waveform_data[pid]['aorta']).rolling(window=10, min_periods=1).mean()).values
        waveform_data[pid]['brach_mean10'] = (pd.Series(waveform_data[pid]['brach']).rolling(window=10, min_periods=1).mean()).values

    ##  Make a feature dictionary
    features = pd.DataFrame()
    features['pid'] = aorta_data[pidstr].values
    if args.testing:
        features['age']=np.nan
    else:
        features['age'] = aorta_data['target'].values

    ## EXPERT FEATURE EXTRACTION
    # Basic feature extraction
    # SBP = Systolic Blood Pressure = max value
    # DBP = Diastolic Blood Pressure = min value
    # PP = pulse pressure

    aDBP = [np.nanmin(waveform_data[pid]['aorta_mean10']) for pid in features['pid']]
    aSBP = [np.nanmax(waveform_data[pid]['aorta_mean10']) for pid in features['pid']]
    bDBP = [np.nanmin(waveform_data[pid]['brach_mean10']) for pid in features['pid']]
    bSBP = [np.nanmax(waveform_data[pid]['brach_mean10']) for pid in features['pid']]
    aPP = [asbp - adbp for asbp, adbp in zip(aSBP, aDBP)]
    bPP = [bsbp - bdbp for bsbp, bdbp in zip(bSBP, bDBP)]

    features['aPP'] = aPP
    features['bDBP'] = bDBP
    features['SBPD'] = np.array(bSBP)-np.array(aSBP)
    features['PP_ratio'] = np.asarray(bPP) / np.asarray(aPP)

    # DATA DRIVEN FEATURE EXTRACTION
    # Fitting overdamped harmonic oscillator to aortic data
    aorta_oharm_A = []
    aorta_oharm_B = []
    aorta_oharm_w0 = []
    for pid in features['pid'].values:
        time_aorta = waveform_data[pid]['time_aorta']
        x = np.array([x[0] for x in time_aorta])
        y = np.array([x[1] for x in time_aorta])
        popt, pcov = curve_fit(overdamped_harm_osc, x, y)
        A, B, w0 = popt
        aorta_oharm_A.append(A)
        aorta_oharm_B.append(B)
        aorta_oharm_w0.append(w0)
        waveform_data[pid]['aorta_oharm_fit'] = overdamped_harm_osc(x, A, B, w0)
    features['aorta_oharm_A'] = aorta_oharm_A
    features['aorta_oharm_B'] = aorta_oharm_B
    features['aorta_oharm_w0'] = aorta_oharm_w0

    # Fitting overdamped harmonic oscillator to brachial data
    brach_oharm_A = []
    brach_oharm_B = []
    brach_oharm_w0 = []
    for pid in features['pid'].values:
        time_brach = waveform_data[pid]['time_brach']
        x = np.array([x[0] for x in time_brach])
        y = np.array([x[1] for x in time_brach])
        popt, pcov = curve_fit(overdamped_harm_osc, x, y)
        A, B, w0 = popt
        #check if it was bad
        if A < 10:
            #use some default values
            brach_oharm_A.append(89.3)
            brach_oharm_B.append(0.8655)
            brach_oharm_w0.append(0.0049)
            waveform_data[pid]['brach_oharm_fit'] = overdamped_harm_osc(x, 89.3, 0.8655, 0.0049)
        else:
            brach_oharm_A.append(A)
            brach_oharm_B.append(B)
            brach_oharm_w0.append(w0)
            waveform_data[pid]['brach_oharm_fit'] = overdamped_harm_osc(x, A, B, w0)

    # Fitting polynomials to the brach residuals from the overdamped harmonic oscillator
    brach_oharm_res_a = []
    brach_oharm_res_b = []
    brach_oharm_res_c = []
    brach_oharm_res_d = []
    brach_oharm_res_e = []
    brach_oharm_res_f = []
    for pid in features['pid'].values:
        time_brach = waveform_data[pid]['time_brach']
        x = np.array([x[0] for x in time_brach])
        y = np.array([x[1] for x in time_brach])
        adata = y - np.array(waveform_data[pid]['brach_oharm_fit'])
        popt, pcov = curve_fit(polynomial5, x, adata)
        a, b, c, d, e, f = popt
        brach_oharm_res_a.append(a)
        brach_oharm_res_b.append(b)
        brach_oharm_res_c.append(c)
        brach_oharm_res_d.append(d)
        brach_oharm_res_e.append(e)
        brach_oharm_res_f.append(f)
        waveform_data[pid]['brach_oharm_res_fit'] = polynomial5(x, a, b, c, d, e, f)
    features['brach_oharm_res_d'] = brach_oharm_res_d
    features['brach_oharm_res_e'] = brach_oharm_res_e

    if args.testing:
        features.to_csv('features_testing.csv', index=False)
        print('done making the features for the testing data')
    else:
        # Save features out to a csv
        features.to_csv('features_training.csv', index=False)
        print('done making the features for the training data')


if __name__ == "__main__":
    main()