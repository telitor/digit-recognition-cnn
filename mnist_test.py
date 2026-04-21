import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import cv2

test_data = dataset.MNIST(
    root = "minst",
    train = False,
    transform = transforms.ToTensor(),
    download = True,
)

testloder = data_utils.DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
)

cnn = torch.load("model/mnist_model.pkl", weights_only=False)
cnn = cnn.cuda()

loss_test = 0
rightValue = 0
loss_func = torch.nn.CrossEntropyLoss();
for index, (images, labels) in enumerate(testloder):

    images = images.cuda()
    labels = labels.cuda()

    outputs = cnn(images)
    _, pred = torch.max(outputs.data, 1)
    loss_test += loss_func(outputs, labels)
    rightValue += (pred == labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in  range(images.shape[0]):
        im_data = images[idx]
        im_data = im_data.transpose(1, 2, 0)
        im_label = labels[idx]
        im_pred = pred[idx]
        print("预测值{}".format(im_pred))
        print("真实值{}".format(im_label))
        cv2.namedWindow("Now_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Now_image", 400, 400)
        cv2.imshow("Now_image", im_data)
        cv2.waitKey(0)





print("loss为{},准确率为{}".format( loss_test,rightValue/len(test_data)))
