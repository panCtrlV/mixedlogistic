import matplotlib.pyplot as plt


def boxplots_c3dxm3dxr3(compiledParameters):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))

    # labels = list([r'$\a_1$', r'$\alpha_{11}$', r'$\alpha_{12}$', r'$\alpha_{13}$'])
    axes[0, 0].boxplot(compiledParameters[:, [0, 2, 3, 4]], showmeans=True)
    axes[0, 0].set_title('Hidden Layer, Group 1')

    axes[0, 1].boxplot(compiledParameters[:, [1, 5, 6, 7]], showmeans=True)
    axes[0, 1].set_title("Hidden Layer, Group 2")

    axes[1, 0].boxplot(compiledParameters[:, [8, 9, 10, 11]], showmeans=True)
    axes[1, 0].set_title('Observed Layer, Group 1')

    axes[1, 1].boxplot(compiledParameters[:, [12, 13, 14, 15]], showmeans=True)
    axes[1, 1].set_title('Observed Layer, Group 2')

    axes[1, 2].boxplot(compiledParameters[:, [16, 17, 18, 19]], showmeans=True)
    axes[1, 2].set_title('Observed Layer, Group 3')

    plt.show()


def boxplot(dataArray):
    plt.figure()
    plt.boxplot(dataArray)
    plt.show()