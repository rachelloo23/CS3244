import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix data
cm = np.array([[493, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [26, 433, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [5, 34, 381, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 451, 54, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 26, 530, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 544, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 19, 0, 0, 0, 3, 0],
               [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 7, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 6],
               [1, 0, 0, 1, 0, 0, 0, 0, 9, 0, 37, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 3, 16]])


# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=True, annot_kws={"size": 12}, linewidths=0.5)

# Set the labels and title
plt.title("Confusion Matrix Heatmap", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()
