import matplotlib.pyplot as plt
import numpy as np

from main import main
from src.postprocess import plotImage
from src.beam_homog_naming import BeamHomog

class Input():
    def __init__(self):
        self.dataset = 'beam_homog'
        self.verbose = True
        self.plot = False
        self.plot_show = False
        self.save_model = False
        self.save_fig = False
        self.model_dir = 'models/manual_data'
        self.fig_dir = 'figures/manual_data'

        # Data parameters
        self.random_test_data = False
        self.random_seed = 1
        self.split_size = 0.1
        self.manual_data = 11

        # Training parameters
        self.epochs = 5000
        self.batch_size = 600
        self.learning_rate = 1e-3
        self.reg = True
        self.reg_coef = 1e-4
        self.code_coef = 1e-3
        self.bias_ord = False
        self.bias_coef = 1

        self.early_stop_patience = 500
        self.early_stop_tol = 0.1

        self.lr_epoch_milestone = [5000]
        self.lr_red_coef = 7e-1

        # Architecture parameters
        self.mode = 'parametric'
        self.layers = [200, 100, 25]
        self.layers_mu = [50, 25]
        self.initialisation = 'kaiming_uniform'

        self.act_code = 'relu'
        self.act_hid = 'relu'
        self.act_out = 'sigmoid'
        self.alpha_relu = 0.5
        self.alpha_sigmoid = 0.5

        self.dropout = False
        self.dropout_prob = 0.1

        # Display parameters
        self.n_disp = 6

def getImageIndex(mus, mus_test):
    for i in range(mus_test.shape[0]):
        if mus == mus_test[i].tolist():
            return i
    else:
        raise RuntimeError('Selected mus not found in test dataset.')

if __name__ == "__main__":
    args = Input()

    # Define test image to be saved
    mus = [2.2, 36]
    args.manual_data = 11
    Fh, Fv, loc, pos = BeamHomog().getParamsFromMus(mus[0],mus[1])

    # Load model and get predicitons
    out, data = main(args)
    
    idx = getImageIndex(mus, data.mus_test)

    # # Compute nodal values for the triangular regular mesh from pixel values
    # nodal_vals = np.zeros([data.resolution[0]+1, data.resolution[1]+1])
    # for i in range(nodal_vals.shape[0]):
    #     for j in range(nodal_vals.shape[1]):
    #         if i == 0 and j==0: # bot-left corner
    #             nodal_vals[i, j] = out[idx][i, j]
    #         elif i == data.resolution[0] and j == 0: # bot-right corner
    #             nodal_vals[i, j] = out[idx][i-1, j]
    #         elif j == data.resolution[1] and i == 0: # top-left corner
    #             nodal_vals[i, j] = out[idx][i, j-1]
    #         elif j == data.resolution[1] and i == data.resolution[0]: # top-right corner
    #             nodal_vals[i, j] = out[idx][i-1, j-1]
    #         elif i == data.resolution[0]: # right boundary:
    #             nodal_vals[i, j] = (out[idx][i-1,j] + out[idx][i-1, j-1])/2
    #         elif i == 0: # left boundary
    #             nodal_vals[i, j] = (out[idx][i,j] + out[idx][i, j-1])/2
    #         elif j == data.resolution[1]: # top boundary
    #             nodal_vals[i, j] = (out[idx][i,j-1] + out[idx][i-1, j-1])/2
    #         elif j == 0: # bot bounary
    #             nodal_vals[i, j] = (out[idx][i,j] + out[idx][i-1, j])/2
    #         else: # node inisde the domain
    #             nodal_vals[i, j] = (out[idx][i,j] + out[idx][i, j-1] + out[idx][i-1,j] + out[idx][i-1, j-1])/4

    # Get one value per trinangular element
    triang_vals = []
    for i in range(data.resolution[0]):
        for j in range(data.resolution[1]):
            triang_vals.append(out[idx][-i,j])
            triang_vals.append(out[idx][-i,j])

    assert len(triang_vals) == data.dimension*2, 'triang_vals shape does not match the correct number'

    # Reshape
    triang_vals = np.reshape(triang_vals, [data.resolution[0]*2, data.resolution[1]], order='F')

    # Plot input image and NN's output (pixel, nodal values and triangular vals)
    plt.subplots(gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})
    plotImage(data.x_test[idx], 3, 1, 1)
    plotImage(out[idx], 3, 1, 2)
    plotImage(triang_vals, 3, 1, 3)
    plt.show()

    # Store prediciton in txt file to be loaded in FreeFEM++
    file_name = f"outputs_txt/mu1_{mus[0]}_mu2_{mus[1]}_Fh_{Fh}_Fv_{Fv}_loc_{loc}_pos_{pos}_case_{args.manual_data}.txt"
    file = open(file_name, "w")
    file.write(f"{triang_vals.size} \n")
    file.write(f"\t")
    count = 1
    for j in range(triang_vals.shape[1]):
        for i in range(triang_vals.shape[0]):
            file.write(f"{triang_vals[i,j]} \t")
            if count % 5 == 0:
                file.write("\n\t")
            count += 1
    file.close()