import numpy as np
import pickle


if __name__ == '__main__':
    with open('../outputs/best_training_case_as_of_19_Feb/4/eval_logs.pkl', 'rb') as f:
        logs = pickle.load(f)

    n_envs = 16
    max_vels = []
    for env in range(1, n_envs+1):
        alphas = logs[f"alphas_ENV_{env}_EPISODE_1"]
        max_vel = np.max(np.abs(np.gradient(alphas, axis=0)))/10.
        max_vels.append(max_vel)

    print(f"Max overall {max(max_vels):.2f} deg/s | Mean of max per episode {np.mean(max_vels):.2f} deg/s")

