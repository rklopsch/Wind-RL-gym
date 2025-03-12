import numpy as  np
import os
import pandas as pd
import pickle
import sys


def split_array_excluding_indices(arr, indices_to_exclude):
    indices_to_exclude = np.sort(np.unique(indices_to_exclude))  # Ensure sorted and unique indices
    split_points = np.where(np.diff(indices_to_exclude) > 1)[0]  # Find gaps in indices
    split_indices = np.split(indices_to_exclude, split_points + 1)  # Split indices into groups

    result = []
    start = 0  # Start of the remaining array

    for group in split_indices:
        end = group[0]  # Start of excluded segment
        if start < end:  # Add valid slice if there's data before excluded indices
            result.append(arr[start:end])
        start = group[-1] + 1  # Move start past the excluded section

    if start < len(arr):  # Add remaining part if exists
        result.append(arr[start:])

    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_individual_powers.py <path to directory containing WindFarm_* directories>")
        sys.exit(1)

    read_indices = False # Set to true if reading previously computed indices 
                         # This is needed if extracting from a Zero case

    directory = sys.argv[1]
    logs = {}
    for env in range(16):
        print(env)
        data = {}
        for turbine in range(3):
            data_file = os.path.join(directory, f'WindFarm_{env+1}', f'disc{turbine + 1}.adm')
            dat = pd.read_csv(data_file, sep="\s+|, ", engine='python')
            # dat['NonDimTime'] = dat['Time'] * self.U / self.D
            data[turbine+1] = dat
            print(np.shape(data[1]['Power_ave']))

        if read_indices:
            reset_idxs = np.load('./sa_sac_array_2/evaluation/long/RL/training_sasac_array_8_eval_2025-02-10_15-46/indices.npy')
            print(np.shape(reset_idxs))

        else:
            # Find out where the yaw angles are 0
            zero_idxs = np.where(data[1]['YawAng'] == 0)[0]

            # Verify that this is consistent across turbines
            for turb in range(2, 4):
                assert np.sum(np.where(data[turb]['YawAng'] == 0)[0] - zero_idxs) == 0
                zero_idxs = np.where(data[turb]['YawAng'] == 0)[0]

            # The reset happens when yaw = 0 and a few steps before that
            # This is because when we call reset, we don't set yaw = 0 immediately
            # But instead we linearly interpolate from the last angle to 0 for some steps
            num_smoothing_steps = 1
            reset_idxs = [zero_idxs[0]]
            for i in range(1, zero_idxs.shape[0]):
                if zero_idxs[i-1] + 1 < zero_idxs[i]:
                    # here we are at the beginning of a new reset, include the smoothing steps
                    lo = max(0, zero_idxs[i] - num_smoothing_steps + 1)
                    hi = zero_idxs[i] + 1
                    reset_idxs += list(range(lo, hi))
                else:
                    # we are somewhere inside a reset
                    reset_idxs.append(int(zero_idxs[i]))
            reset_idxs = np.asarray(reset_idxs)
            print(np.shape(reset_idxs))

        # Now we split this into episodes
        # reset_idxs now has the inices where we are inside a reset
        powers = []
        for turb in range(1, 4):
            arr = list(data[turb]['Power_ave'])
            split_result = split_array_excluding_indices(arr, reset_idxs)
            powers.append(split_result)

        powers = np.asarray(powers)  # should be n_turbs x n_episodes x n_steps_per_ep
        print(powers.shape)
        powers = np.transpose(powers, (1, 2, 0))  # now n_episodes x n_steps x n_turbs
        powers /= 1e6

        if not powers.shape[1] == 2000:
            raise Warning(f"Episode length is {powers.shape[1]}")

        for ep in range(powers.shape[0]):
            logs[f"powers_ENV_{env+1}_EPISODE_{ep+1}"] = powers[ep, :, :]

    print(f"Info: Turbine powers are already in MW.")

    filename = os.path.join(directory, f'turbine_powers.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(logs, f)

    if not read_indices:
        np.save(os.path.join(directory, 'indices.npy'), reset_idxs)

    print(f"Saved file to {os.path.abspath(filename)}")


