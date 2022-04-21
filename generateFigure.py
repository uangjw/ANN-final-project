import matplotlib.pyplot as plt

epochs = list(range(0, 50))
train_loss = []
val_loss = []
train_acc = []
val_acc = []
file = open('method1_result/fold5.txt')
i = 1
for line in file:
    if i % 3 == 1:  # train loss & acc
        train_loss.append(float(line[0:6]))
        train_acc.append(float(line[8:14]))
    elif i % 3 == 2:    # val loss & acc
        val_loss.append(float(line[0:6]))
        val_acc.append(float(line[8:14]))
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