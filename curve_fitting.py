import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

file_path = 'amplitude_separatrix_crossing_time.xlsx'

df = pd.read_excel(file_path)

amplitudes = df[df.columns[0]].values
crossing_times = df[df.columns[1]].values

def power_law(amplitude, a, b):
    return a* amplitude**(-b)

def exponential_decay(amplitude, a, b):
    return a*np.exp(-b*amplitude)

def inverse_model(amplitude, a, b):
    return a/ (amplitude + b)

#Fitting power law, exponential decay and inverse function model

#Power law
params_power, _ = curve_fit(power_law, amplitudes, crossing_times)
crossing_time_power_law = power_law(amplitudes, *params_power)
a_power, b_power = params_power

#Exponential decay
params_exp, _ = curve_fit(exponential_decay, amplitudes, crossing_times)
crossing_time_exp = exponential_decay(amplitudes, *params_exp)
a_exp, b_exp = params_exp

#Inverse function model
params_inv, _ = curve_fit(inverse_model, amplitudes, crossing_times)
crossing_time_inv = inverse_model(amplitudes, *params_inv)
a_inv, b_inv = params_inv

print(f'Power law model: Crossing Time = {a_power:.3f} * Amplitude^(-{b_power:.3f})\n')
print(f'Exponential decay: Crossing time = {a_exp:.3f} * exp(-{b_exp:.3f} * Amplitude)\n')
print(f'Inverse model: Crossing time = {a_inv:.3f} / (Amplitude + {b_inv:.3f})\n')


#Error metric functions

def mean_absolute_error(observed, predicted):
    return np.mean(np.abs(observed - predicted))

def root_mean_squared_error(observed, predicted):
    return np.sqrt(np.mean((observed - predicted)**2))


#Calculate errors
mae_power = mean_absolute_error(crossing_times, crossing_time_power_law)
rmse_power = root_mean_squared_error(crossing_times, crossing_time_power_law)

mae_exp = mean_absolute_error(crossing_times, crossing_time_exp)
rmse_exp = root_mean_squared_error(crossing_times, crossing_time_exp)

mae_inv = mean_absolute_error(crossing_times, crossing_time_inv)
rmse_inv = root_mean_squared_error(crossing_times, crossing_time_inv)

error_metrics = {
    'Model': ['Power law', 'Exponential decay', 'Inverse function'],
    'MAE': [mae_power, mae_exp, mae_inv],
    'RMSE': [rmse_power, rmse_exp, rmse_inv]
}

print(error_metrics)