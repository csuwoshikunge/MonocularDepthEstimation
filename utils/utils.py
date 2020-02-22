import matplotlib.pyplot as plt

def saveplot(img, name):
    ii = plt.imshow(img, interpolation='nearest')
    plt.set_cmap('viridis')
    plt.colorbar(ii)
    plt.savefig(name)
    plt.clf()