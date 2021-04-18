import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('temp/dist_normal.pkl', 'rb') as f:
    dist_normal = pickle.load(f)
with open('temp/dist_seizure.pkl', 'rb') as f:
    dist_seizure = pickle.load(f)

print(len(dist_normal))
print(len(dist_seizure))
print("Normal samples: Max->", max(dist_normal),
      "Min->", min(dist_normal), 
      "Avg->", sum(dist_normal)/len(dist_normal))
print("Seizure samples: Max->", max(dist_seizure),
      "Min->", min(dist_seizure), 
      "Avg->", sum(dist_seizure)/len(dist_seizure))

plt.figure(figsize=(20,7))
plt.title("L2 reconstruction distance")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.plot(dist_normal,label="Normal periods")
plt.plot(dist_seizure,label="Seizure periods")
plt.legend()
plt.show()
#plt.savefig("reconstruction_dist.jpg")
