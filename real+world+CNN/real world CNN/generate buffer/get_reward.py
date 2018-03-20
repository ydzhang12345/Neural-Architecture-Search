import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt

moving_acc = []
total_acc = []
moving_reward = []
total_reward = []
nn = []

n = 0
with open('./train_history.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		if n==0:
			n += 1
			continue
		nn.append(n)
		n += 1
		acc, reward, _, _, _ = row
		total_acc.append(float(acc))
		total_reward.append(float(reward)*100)
		moving_acc.append(np.mean(total_acc[-20:]))
		moving_reward.append(np.mean(total_reward[-250:]))



'''
plt.plot(nn[50:], moving_reward[50:], 'r')
plt.ylabel('Average Reward')
plt.xlabel('Number of Iterations')
plt.title('Average Reward in DNN-KWS')
#plt.show()
plt.savefig('./milestone_reward.jpg')
'''

plt.plot(nn[50:], moving_acc[50:], 'r')
plt.ylabel('Average Acc')
plt.xlabel('Number of Iterations')
plt.title('Average Accuracy in DNN-KWS')
plt.show()
#plt.savefig('./milestone_reward.jpg')
