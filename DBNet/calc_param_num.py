from networks.dinknet import ResNet34_EdgeNet, ResNet34_BRN, ResNet34_


net = ResNet34_EdgeNet()
params = list(net.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和(Byte)：" + str(k))
