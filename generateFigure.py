import matplotlib.pyplot as plt

epochs = list(range(0, 200))
train_loss = []
val_loss = []
train_acc = []
val_acc = []
file = open('result/result-vitb1-200epoch-mixup.txt')
i = 0
for line in file:
    if i % 4 == 2:  # train loss & acc
        train_loss.append(float(line[12:18]))
        train_acc.append(float(line[24:30]))
    elif i % 4 == 3:    # val loss & acc
        val_loss.append(float(line[10:16]))
        val_acc.append(float(line[22:28]))
    i += 1
file.close()

plt.figure(1)
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="val")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()

plt.figure(2)
plt.plot(epochs, train_acc, label="train")
plt.plot(epochs, val_acc, label="val")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim((0.0, 0.8))
plt.legend()
plt.show()