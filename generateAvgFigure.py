import matplotlib.pyplot as plt

epochs = list(range(0, 50))
train_loss = [0 for i in range(50)]
val_loss = [0 for i in range(50)]
train_acc = [0 for i in range(50)]
val_acc = [0 for i in range(50)]
file = []
file.append(open('method1_result/fold1.txt'))
file.append(open('method1_result/fold2.txt'))
file.append(open('method1_result/fold3.txt'))
file.append(open('method1_result/fold4.txt'))
file.append(open('method1_result/fold5.txt'))

for j in range(5):
    i = 1
    id = 0
    for line in file[j]:
        if i % 3 == 1:  # train loss & acc
            train_loss[id] += float(line[0:6])
            train_acc[id] += float(line[8:14])
        elif i % 3 == 2:    # val loss & acc
            val_loss[id] += float(line[0:6])
            val_acc[id] += float(line[8:14])
            id += 1
        i += 1
    file[j].close()

for k in range(50):
    train_loss[k] /= 5
    train_acc[k] /= 5
    val_loss[k] /= 5
    val_acc[k] /= 5

plt.figure(1)
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="val")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("5-fold cross-validation")
plt.legend()

plt.figure(2)
plt.plot(epochs, train_acc, label="train")
plt.plot(epochs, val_acc, label="val")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("5-fold cross-validation")
plt.ylim((0.0, 0.8))
plt.legend()
plt.show()