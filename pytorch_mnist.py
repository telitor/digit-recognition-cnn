import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
###数据加载

train_data = dataset.MNIST(
    root = "minst",
    train = True,
    transform = transforms.ToTensor(),
    download = True,
)
test_data = dataset.MNIST(
    root = "minst",
    train = False,
    transform = transforms.ToTensor(),
    download = True,
)

###分批加载

trainloder = data_utils.DataLoader(
    dataset = train_data,
    batch_size = 64,
    shuffle = True,
)
testloder = data_utils.DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
)

cnn = CNN();

cnn = cnn.cuda();

###损失函数

loss_func = torch.nn.CrossEntropyLoss();

###优化函数

optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.01);

###训练
for epoch in range(10):

    ###测试
    for index, (images, labels) in enumerate(trainloder):

        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)

        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("当前为第{}轮，批次为{}/{},loss为{}".format(epoch+1,index+1,len(train_data)//64+1, loss.item()))

    ###验证
    loss_test = 0
    rightValue = 0
    for index, (images, labels) in enumerate(testloder):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)

        loss = loss_func(outputs, labels)

        loss_test += loss_func(outputs,labels)
        _,pred = outputs.max(1)

        rightValue += (pred == labels).sum().item()
        print("当前为第{}轮的测试集验证，批次为{}/{},loss为{},准确率为{}".format(epoch + 1, index + 1, len(test_data) // 64 + 1, loss.item(),rightValue/len(test_data)))

torch.save(cnn,"model/mnist_model.pkl")