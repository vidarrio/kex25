
import json
import matplotlib.pyplot as plt

# Path to the JSON file with training metrics
metrics_file = '/home/ubuntu/kex25/src/models/v1.2.1.pth_training_metrics__20250416-151004.json'

# Load the metrics from the JSON file
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

avg_losses = metrics['avg_losses']
avg_score = metrics['avg_score']

# Create the plots
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(avg_losses, label="Avg Q Loss", color='blue')
plt.title("Average Q Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(avg_score, label="Avg Score", color='orange')
plt.title("Average Score per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()

# Save the plot as a PNG file instead of displaying it
output_file = '/home/ubuntu/kex25/src/models/plot.png'
plt.savefig(output_file)
plt.close()

print(f"Plot saved to {output_file}")