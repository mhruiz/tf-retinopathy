from . import evaluation as ev

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


# Plotting function for training metrics evolution
def plot_metric(history, metric_name):
    plt.plot(history.history[metric_name], label='Train ' + metric_name)
    try:
        plt.plot(history.history['val_' + metric_name], label='Validation ' + metric_name)
        plt.title('Train vs Val: ' + metric_name)
    except KeyError:
        plt.title(metric_name + ' evolution on training')
    plt.legend(loc='best')
    plt.show()

from importlib import reload
reload(ev)

def plot_roc_curve(fpr, tpr, thresholds, dataset_name, ROC_names, operative_threshold=None, grid=False, show_points=False, show_zoomed=True):

    # Convert arguments into lists if thery are not
    if type(fpr) != list:
        fpr = [fpr]
    if type(tpr) != list:
        tpr = [tpr]
    if type(thresholds) != list:
        thresholds = [thresholds]
    if type(ROC_names) != list:
        ROC_names = [ROC_names]
    if type(operative_threshold) != list and type(operative_threshold) != tuple:
        operative_threshold = [operative_threshold] * 3

    points_colors = ['ko', 'bo', 'ro', 'go', 'yo']

    # These are the closests points
    closest_points = []

    plt.figure()

    if show_zoomed:
        plt.subplots(1,2, figsize=(24,10))
        plt.subplot(121)

    # Plot ROC curve (all)
    plt.plot([0, 1], [0, 1], 'k--')

    for i in range(len(fpr)):

        # Get current ROC
        fpr_i = fpr[i]
        tpr_i = tpr[i]
        thresholds_i = thresholds[i]
        operative_threshold_i = operative_threshold[i]

        if operative_threshold_i is None:
            # Find closest point to Up-Left corner (with interpolation)
            fpr_, tpr_, thr_ = ev.find_closest_point_to_01(fpr_i, tpr_i, thresholds_i)

        else:
            # Get fpr and tpr by interpolation
            fpr_, tpr_ = ev.get_fpr_tpr_at_threshold(fpr_i, tpr_i, thresholds_i, operative_threshold_i)

            # Get fpr and tpr using closest threshold
            # thr_index = ev.find_closest_value(thresholds, operative_th, ascending=False)

            # thr_index = thr_index if np.abs(thresholds[thr_inx] - operative_th) < np.abs(thresholds[thr_inx - 1] - operative_th) else thr_index - 1

            # fpr_ = fpr[thr_index]
            # tpr_ = tpr[thr_index]

        closest_point = (fpr_, tpr_)

        closest_points.append(closest_point)

        plt.plot(fpr_i, tpr_i, label=ROC_names[i] +' - AUC = {:.3f}'.format(ev.get_auc(fpr_i, tpr_i)))
        if show_points:
            plt.plot(fpr_i, tpr_i, 'o')

        # Operative point
        plt.plot(closest_point[0], closest_point[1], points_colors[i % len(points_colors)], 
                 label='Op. point ' + ROC_names[i] + ' - Sens: ' +\
                       str(round(closest_point[1]*100,2)) + ', Sp: ' +\
                       str(round((1-closest_point[0])*100,2)))
        
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(dataset_name)
    plt.legend(loc='best')

    plt.grid(grid)

    if show_zoomed:
        # Zoom in view of the upper left corner.
        plt.subplot(122)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot([0, 1], [0, 1], 'k--')

        for i in range(len(fpr)):
        
            # Get current ROC
            fpr_i = fpr[i]
            tpr_i = tpr[i]

            plt.plot(fpr_i, tpr_i, label=ROC_names[i] +' - AUC = {:.3f}'.format(ev.get_auc(fpr_i, tpr_i)))
            if show_points:
                plt.plot(fpr_i, tpr_i, 'o')

            # Operative point
            plt.plot(closest_points[i][0], closest_points[i][1], points_colors[i % len(points_colors)], 
                    label='Op. point ' + ROC_names[i] + ' - Sens: ' +\
                        str(round(closest_points[i][1]*100,2)) + ', Sp: ' +\
                        str(round((1-closest_points[i][0])*100,2)))
            
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        # plt.title(dataset_name + ' - ROC curve (zoomed in at top left). \n' +\
        #           'Closest point: ' + str(round((1-closest_point[0]) * 100, 2)) + '% Specifity - ' + str(round(closest_point[1] * 100, 2)) + '% Sensibility')
        plt.title(dataset_name)
        plt.legend(loc='best')
        
        plt.grid(grid)
    
    plt.show()


def plot_confusion_metric(confusion_matrix, dr_lvls_per_box, DR_LEVELS_PER_CLASS, title=None):

    matrix_flatten = [int(x) for x in confusion_matrix.flatten().tolist()]

    num_all_elements = np.sum(matrix_flatten)

    def get_box_name(num_elements_in_box, dr_lvls_inside):
        
        box_name = 'Num. images: ' + str(num_elements_in_box)# + ' -- (' +  str(round((num_elements_in_box/num_all_elements)*100)) + ' %)'
            
        if dr_lvls_inside != {}:
            for k in dr_lvls_inside:
                box_name += '\nDR level ' + str(k) + ': ' + str(dr_lvls_inside[k])
        
        return box_name

    boxes_names = [get_box_name(num_imgs, dr_per_box) for num_imgs, dr_per_box in zip(matrix_flatten, dr_lvls_per_box)]

    boxes_names = np.asarray(boxes_names).reshape(len(DR_LEVELS_PER_CLASS), len(DR_LEVELS_PER_CLASS))

    axis_names = ['Class ' + str(class_) + ' -> DR Lvls: ' + \
        ','.join(list(map(str, list(DR_LEVELS_PER_CLASS[class_])))) \
            for class_ in range(len(DR_LEVELS_PER_CLASS))]
    
    row_names = ['Class ' + str(class_) for class_ in range(len(DR_LEVELS_PER_CLASS))]

    df_cm = pd.DataFrame(confusion_matrix, index = row_names, columns = axis_names)

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=boxes_names, cmap='Blues', fmt='')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    if title is None:
        plt.title('Confusion matrix')
    else:
        plt.title('Confusion matrix - ' + title)
    plt.show()