
def plot_histogram(A):
    A=A.flatten()
    plt.hist(A, bins=100)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()