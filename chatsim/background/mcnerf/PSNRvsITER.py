'''
import re
import matplotlib.pyplot as plt

# Use a non-GUI backend for cluster environments
import matplotlib
matplotlib.use('Agg')  # Prevents need for a display

# Path to your MCNeRF log file
log_path = "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/156453.out"  # Replace with the actual path

iterations = []
psnr_values = []

with open(log_path, 'r') as f:
    for line in f:
        match = re.search(r"Iter:\s+(\d+)\s+PSNR:\s+([\d\.]+)", line)
        if match:
            iter_val = int(match.group(1))
            psnr_val = float(match.group(2))
            iterations.append(iter_val)
            psnr_values.append(psnr_val)

# Plot and save
plt.figure(figsize=(10, 5))
plt.plot(iterations, psnr_values, marker='o', markersize = 1.8,  linewidth=1.5, label='PSNR')
plt.title("PSNR over Iterations (MCNeRF Training)")
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot to a file
output_path = "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/psnr_plot.png"  # Change to .pdf if preferred
plt.savefig(output_path, dpi=300)

print(f"[✅] Plot saved to: {output_path}")
'''

###############################################################
###############################################################

import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For clusters (no GUI)

# List your log files here with a label
log_files = {
    "1137": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/147941.out",
    "1006": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/153748.out",
    "4058": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/156453.out",
    "1137+cyclegan_1": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/157725.out"
    #"1137+cyclegan_2": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/158255.out"
}

# Store results
psnr_data = {}

for label, path in log_files.items():
    iterations = []
    psnr_values = []
    with open(path, 'r') as f:
        for line in f:
            match = re.search(r"Iter:\s+(\d+)\s+PSNR:\s+([\d\.]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                psnr_values.append(float(match.group(2)))
    psnr_data[label] = (iterations, psnr_values)
    print(f"[OK] Parsed {len(iterations)} points from {label}")

# Plot
plt.figure(figsize=(12, 6))
for label, (iters, psnrs) in psnr_data.items():
    plt.plot(iters, psnrs, label=label, linewidth=1.5, markersize=2, marker='o')

plt.title("PSNR Comparison Across Experiments")
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend(loc= "lower right")
plt.tight_layout()

# Save plot
output_path = "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/compare_psnr_runs_1.png"
plt.savefig(output_path, dpi=300)
print("[✅] Saved as compare_psnr_runs_1.png")

#################################################
####################################################

# import re
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # For non-GUI cluster use

# # Define log files with labels
# log_files = {
#     "1137": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/158257.out",
#     "1137+cyclegan": "/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/logs/158255.out"
# }

# # Store results
# mse_data = {}
# lr_data = {}

# # Parse logs
# for label, path in log_files.items():
#     iterations = []
#     mse_values = []
#     lr_values = []
#     with open(path, 'r') as f:
#         for line in f:
#             match = re.search(r"Iter:\s+(\d+)\s+PSNR:\s+[\d\.]+\s+MSE:\s+([\d\.]+).*?LR:\s+([\d\.]+)", line)
#             if match:
#                 iterations.append(int(match.group(1)))
#                 mse_values.append(float(match.group(2)))
#                 lr_values.append(float(match.group(3)))
#     mse_data[label] = (iterations, mse_values)
#     lr_data[label] = (iterations, lr_values)
#     print(f"[OK] Parsed {len(iterations)} entries from {label}")

# # Plot MSE
# plt.figure(figsize=(10, 5))
# for label, (iters, mses) in mse_data.items():
#     plt.plot(iters, mses, label=label, linewidth=1.5, marker='o', markersize=2)
    
# plt.yscale("log")
# plt.title("MSE Loss Comparison")
# plt.xlabel("Iteration")
# plt.ylabel("MSE Loss(log scale)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/compare_mse_plot.png", dpi=300)
# print("[✅] Saved MSE plot as compare_mse_plot.png")
# plt.clf()

# # Plot LR
# plt.figure(figsize=(10, 5))
# for label, (iters, lrs) in lr_data.items():
#     plt.plot(iters, lrs, label=label, linewidth=1.5, linestyle='--')
    
# plt.title("Learning Rate Schedule")
# plt.xlabel("Iteration")
# plt.ylabel("Learning Rate")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/compare_lr_plot.png", dpi=300)
# print("[✅] Saved LR plot as compare_lr_plot.png")
