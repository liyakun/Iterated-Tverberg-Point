import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltpage


class Plot:

    def __init__(self):
        pass

    def plot(self, weights_all):
        #size = len(weights_all)
        with pltpage.PdfPages("../resources/pic/weights.pdf") as pdf:
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
            plt.plot(weights_all)
            pdf.savefig(fig)
            plt.close()

