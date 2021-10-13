from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np

# Evaluation functions for ROC curve
# True positives, false positives, true negatives and false negatives
def get_tp_fp_tn_fn(y_true, y_pred):
    TP = FP = TN = FN = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == y_p:
            if y_t == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_t == 1:
                FN += 1
            else:
                FP += 1
    return TP, FP, TN, FN


# Optimized funciton to obtain false and true positive rates at a given threshold (which belongs to a sorted )
def get_fpr_tpr_at_threshold_optimized(y_true, y_pred, threshold, previous_which_are_0=None, TP_ant=None, FP_ant=None, TN_ant=None, FN_ant=None):
    # Model output with threshold applied -- all ones
    thresholdized = np.ones((y_pred.shape[0],), dtype=np.uint8)

    # Due to thresholds are given from higher to lower, we only want to get all the values that were lower than previous thresholds
    # If there is no previous vector, this is the first threshold, every position needs to be calculated
    check_positions = previous_which_are_0
    if check_positions is None:
        # Every position
        check_positions = list(range(len(y_pred)))

    TP = FP = TN = FN = 0

    still_are_0 = []

    for i in check_positions:
    
        if y_pred[i] <= threshold:
            thresholdized[i] = 0
            # Save positions of elements whose value is still '0' for next
            still_are_0.append(i)

        if y_true[i] == thresholdized[i]:
            if y_true[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_true[i] == 1:
                FN += 1
            else:
                FP += 1
    
    if previous_which_are_0 is not None:
        # Substract from known TN on previous array every False Positive detected now
        TN = TN_ant - FP
        # Substract from known FN on previous array every True Positive detected now
        FN = FN_ant - TP
        # Add every positive detected (both True and False)
        TP += TP_ant
        FP += FP_ant
    
    fpr = (FP / (FP + TN)) if (FP + TN) != 0 else 0
    tpr = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    
    return fpr, tpr, still_are_0, TP, FP, TN, FN, 


def get_joined_array(array, class_):

    '''
    Combine into a single array the columns of 'array' specified by 'class_'
    '''

    if type(class_) == tuple:
        new_array = np.zeros((array.shape[0],))

        # Join columns by adding
        for j in class_:
            new_array = new_array + array[:,j]
    else:
        new_array = array[:, class_]
    
    return new_array


# Own roc curve, with possibility to specify a initial number of thresholds
def get_roc_curve(y_true_all, y_pred_all, classes=None, num_thresholds=None, remove_intermediate_points=True, remove_under_ROC_points=True):

    # Returns multiple ROCs in lists
    FPR_list = []
    TPR_list = []
    thresholds_list = []

    if y_true_all.ndim == 1:
        # Only one class, create a 2nd dimmension, just for the loop
        y_true_all = np.expand_dims(y_true_all, axis=1)
        y_pred_all = np.expand_dims(y_pred_all, axis=1)
    
    # Classes must be a list of integers or tuples, that will specify which columns have to be joined to get its ROC
    if classes is None:
        classes = list(range(y_pred_all.shape[1]))
    elif type(classes) != list:
        classes = [classes]

    for class_ in classes:

        y_true = get_joined_array(y_true_all, class_)
        y_pred = get_joined_array(y_pred_all, class_)

        # Get thresholds
        if num_thresholds is None:
            # Use as threshold every value in predicted array (and add 0 and 1 if they were not inside)
            thresholds = np.sort(np.unique(np.append(np.copy(y_pred), [0, 1]))) # add 0 and 1 in case that they do not exist

        else:
            # These are all the thresholds
            thresholds = np.linspace(0, 1, num_thresholds)

        # Higher to lower --> This will make that FPR and TPR will be ascending sorted
        thresholds = thresholds[::-1]

        TPR = np.zeros((thresholds.shape[0]))
        FPR = np.zeros((thresholds.shape[0]))

        ignored_points = []

        still_are_0 = None
        tp = fp = tn = fn = None

        # From higher to lower thresholds
        for i, th in enumerate(thresholds):
            fpr, tpr, still_are_0, tp, fp, tn, fn = get_fpr_tpr_at_threshold_optimized(y_true, y_pred, th, still_are_0, tp, fp, tn, fn)

            # tp, fp, tn, fn = get_tp_fp_tn_fn(y_true, y_pred > th)
            # tpr = tp / (tp + fn)
            # fpr = fp / (fp + tn)

            # Ignore intermediate points
            # Assuming that there must not be any point whose TPR is lower than the previous point
            if i > 0:
                # If same false positive rate than previous point (same x in ROC), ignore point whose TPR is lower (the previous one)
                if FPR[i-1] == fpr:
                    if i > 1:
                        ignored_points.append(i-1)
                # If same true positive rate than previous point (same y in ROC), ignore point whose FPR is higher (current one)
                elif TPR[i-1] == tpr:
                    if i < len(thresholds) - 1:
                        ignored_points.append(i)

            TPR[i] = tpr
            FPR[i] = fpr

        # First threshold (1) must have always 0 fpr and 0 tpr
        # Last threshold (0) must have always 1 fpr and 1 tpr
        FPR[0] = 0
        TPR[0] = 0
        FPR[-1] = 1
        TPR[-1] = 1

        # Drop intermediate
        if remove_intermediate_points:
            TPR = np.delete(TPR, ignored_points)
            FPR = np.delete(FPR, ignored_points)
            thresholds = np.delete(thresholds, ignored_points)

        for i in range(len(FPR)):

            fpr = FPR[i]
            tpr = TPR[i]
            th = thresholds[i]

        if remove_under_ROC_points:

            def get_sign_vectorial_product(a, b):
                # If positive -> anticlockwise; else -> clockwise
                return a[0] * b[1] - a[1] * b[0]
            
            correct_points = np.ones((TPR.shape[0]), dtype=np.bool)

            for i in range(TPR.shape[0] - 2):
                # If current point was discarded, continue to next
                if not correct_points[i]:
                    continue

                for k in range(1, TPR.shape[0]-1-i):
                    # Current vector = i to k+i
                    current_vector = (FPR[i+k] - FPR[i], TPR[i+k] - TPR[i])

                    # Next vectors, from i to k+i[+1 +2 +3 ...]
                    for j in range(i+k+1, TPR.shape[0]):
                        # If next point was discarded, continue to next
                        if not correct_points[j]:
                            continue
                        next_vector = (FPR[j] - FPR[i], TPR[j] - TPR[i])
                        
                        sign = get_sign_vectorial_product(current_vector, next_vector)

                        if sign >= 0: # Positive value means anticlockwise, so this point must be discarded
                            correct_points[i+k] = False
                            break
            
            # Drop discarded points
            FPR = FPR[correct_points]
            TPR = TPR[correct_points]
            thresholds = thresholds[correct_points]

        FPR_list.append(FPR)
        TPR_list.append(TPR)
        thresholds_list.append(thresholds)

    return FPR_list, TPR_list, thresholds_list


def get_interpolated_threshold(higher_fpr, lower_fpr, higher_tpr, lower_tpr, higher_thr, lower_thr, interpolated_fpr, interpolated_tpr):
    
    # Distance between ROC points, defined by fpr and tpr values
    points_ROC_distance = np.sqrt((higher_fpr - lower_fpr)**2 + (higher_tpr - lower_tpr)**2)

    # Distante between lower ROC point to the given interpolated ROC point
    distance_to_interpolated = np.sqrt((interpolated_fpr - lower_fpr)**2 + (interpolated_tpr - lower_tpr)**2)

    # Distance between the 2 thresholds
    thresholds_distance = higher_thr - lower_thr

    # Interpolate threshold
    interpolated_thr = higher_thr - (distance_to_interpolated / points_ROC_distance) * thresholds_distance

    return interpolated_thr


def get_closest_point_to_01_in_pairs(fpr, tpr, thresholds):
    '''
    This function will work as a generator.
    It will take every pair of points in the ROC and return the closest point to 01
    This point can be one of these two points or an interpolated one between them
    '''

    # Up - Left corner
    corner = np.array([0, 1])

    for point in range(len(fpr) - 1):

        point_1 = (fpr[point], tpr[point])
        point_2 = (fpr[point+1], tpr[point+1])

        dist_point_1 = np.linalg.norm(np.array(point_1) - corner)
        dist_point_2 = np.linalg.norm(np.array(point_2) - corner)

        # min_1 --> closest value, min_2 --> second closest (in current pair of points)
        if dist_point_1 < dist_point_2:
            min_1_pos, min_2_pos = point, point + 1
            min_dist = dist_point_1
        else:
            min_1_pos, min_2_pos = point + 1, point
            min_dist = dist_point_2

        # Get vector between the two points
        # Its sign must be (+, +), so vector must be calculated as: (higher fpr - lower fpr, higher tpr - lower tpr)
        # fpr and tpr were obtained using a descending sorted thresholds array, so thr[0] = 1 --> fpr[0] = tpr[0] = 0
        # higher position means higher values of fpr and tpr
        higher_pos, lower_pos = (min_1_pos, min_2_pos) if min_1_pos >= min_2_pos else (min_2_pos, min_1_pos)

        # Vector
        v = (fpr[higher_pos] - fpr[lower_pos], tpr[higher_pos] - tpr[lower_pos])

        # Perpendicular (normal) vector --> (+, -)
        n = (v[1], -v[0])

        if v[0] == 0 or n[0] == 0:
            # This means these 2 points have the same coordinate on X or Y, just returns the closest one
            yield fpr[min_1_pos], tpr[min_1_pos], thresholds[min_1_pos], min_dist
            continue

        # Rect
        # (x0, y0) + k·v = (x, y)
        # y = mx + n ---> m = vy / vx
        #                 n = y0 - mx0
        # y = mx + n ---> mx - y = -n ---> Ax + By = C
        #  A = m = vy / vx
        #  B = -1
        #  C = -n = -(y0 - mx0)

        A_v = v[1] / v[0] # m
        B_v = -1
        C_v = -(tpr[lower_pos] - A_v * fpr[lower_pos])

        A_n = n[1] / n[0] # m
        B_n = -1
        C_n = -(1 - A_n * 0) # (x0, y0) = up left corner = (0, 1)

        # Get intersection point by applying Cramer
        # { A_v·x B_v·y = C_v
        # { A_n·x B_n·y = C_n
        # 
        # | A_v B_v : C_v |
        # | A_n B_n : C_n |

        # x = det(C,B) / det(A,B)
        # y = det(A,C) / det(A,B)

        det_AB = A_v * B_n - A_n * B_v
        det_CB = C_v * B_n - C_n * B_v
        det_AC = A_v * C_n - A_n * C_v

        # Intersection point
        fpr_intersect = det_CB / det_AB
        tpr_intersect = det_AC / det_AB 

        # Check if intersection point is truly the closest point to (0, 1)
        # If the intersection point is not between min_1 and min_2, the closest point (of this pair) will be min_1
        if fpr_intersect < fpr[lower_pos] or fpr_intersect > fpr[higher_pos]:
            # Min_1 is the closest point, return it
            yield fpr[min_1_pos], tpr[min_1_pos], thresholds[min_1_pos], min_dist

        else:
            # Get interpolated threshold
            thr_intersect = get_interpolated_threshold(fpr[higher_pos], 
                                                    fpr[lower_pos], 
                                                    tpr[higher_pos], 
                                                    tpr[lower_pos], 
                                                    thresholds[lower_pos], 
                                                    thresholds[higher_pos], 
                                                    fpr_intersect, 
                                                    tpr_intersect)

            yield fpr_intersect, tpr_intersect, thr_intersect, np.linalg.norm(np.array([fpr_intersect, tpr_intersect]) - corner)


def find_closest_point_to_01(fpr, tpr, thresholds):

    candidates = [x for x in get_closest_point_to_01_in_pairs(fpr, tpr, thresholds)]

    min_index = np.argmin([x[-1] for x in candidates if x != np.nan])

    return candidates[min_index][:-1]

'''
def find_closest_point_to_01(fpr, tpr, thresholds):
    min_1_dist = min_2_dist = np.Inf
    min_1_pos = min_2_pos = -1

    # min_1 --> minimum value, min_2 --> second minimum

    # Up - Left corner
    corner = np.array([0, 1])
    
    for i, (fpr_i, tpr_i) in enumerate(zip(fpr, tpr)):
        # Get distance
        current_dist = np.linalg.norm(np.array([fpr_i, tpr_i]) - corner)
        
        if min_1_dist > current_dist:
            # Replace second minimum with min_1
            min_2_dist = min_1_dist
            min_2_pos = min_1_pos

            # Replace minimum
            min_1_dist = current_dist
            min_1_pos = i
        elif min_2_dist > current_dist:
            min_2_dist = current_dist
            min_2_pos = i

    # Get vector between the two closest points to (0, 1)
    # Its sign must be (+, +), so vector must be calculated as: (higher fpr - lower fpr, higher tpr - lower tpr)
    # fpr and tpr were obtained using a descending sorted thresholds array, so thr[0] = 1 --> fpr[0] = tpr[0] = 0
    # higher position means higher values of fpr and tpr
    higher_pos, lower_pos = (min_1_pos, min_2_pos) if min_1_pos >= min_2_pos else (min_2_pos, min_1_pos)

    # Vector
    v = (fpr[higher_pos] - fpr[lower_pos], tpr[higher_pos] - tpr[lower_pos])

    # Perpendicular (normal) vector --> (+, -)
    n = (v[1], -v[0])

    # Rect
    # (x0, y0) + k·v = (x, y)
    # y = mx + n ---> m = vy / vx
    #                 n = y0 - mx0
    # y = mx + n ---> mx - y = -n ---> Ax + By = C
    #  A = m = vy / vx
    #  B = -1
    #  C = -n = -(y0 - mx0)

    A_v = v[1] / v[0] # m
    B_v = -1
    C_v = -(tpr[lower_pos] - A_v * fpr[lower_pos])

    A_n = n[1] / n[0] # m
    B_n = -1
    C_n = -(1 - A_n * 0) # (x0, y0) = up left corner = (0, 1)

    # Get intersection point by applying Cramer
    # { A_v·x B_v·y = C_v
    # { A_n·x B_n·y = C_n
    # 
    # | A_v B_v : C_v |
    # | A_n B_n : C_n |

    # x = det(C,B) / det(A,B)
    # y = det(A,C) / det(A,B)

    det_AB = A_v * B_n - A_n * B_v
    det_CB = C_v * B_n - C_n * B_v
    det_AC = A_v * C_n - A_n * C_v

    # Intersection point
    fpr_intersect = det_CB / det_AB
    tpr_intersect = det_AC / det_AB 

    # Check if intersection point is truly the closest point to (0, 1)
    # If the intersection point is not between min_1 and min_2, the closest point will be min_1
    if fpr_intersect < fpr[lower_pos] or fpr_intersect > fpr[higher_pos]:
        # Min_1 is the closest point, return it
        return fpr[min_1_pos], tpr[min_1_pos], thresholds[min_1_pos]

    # Get interpolated threshold
    thr_intersect = get_interpolated_threshold(fpr[higher_pos], 
                                               fpr[lower_pos], 
                                               tpr[higher_pos], 
                                               tpr[lower_pos], 
                                               thresholds[lower_pos], 
                                               thresholds[higher_pos], 
                                               fpr_intersect, 
                                               tpr_intersect)

    return fpr_intersect, tpr_intersect, thr_intersect
'''

def find_closest_value(thresholds, thr, ascending=True):

    '''
    Binary search function. Given an array and a value, it will search the first index in the array whose element
    is higher or lower than the given value.

    thresholds:
        1-D numpy array.
    thr:
        value to search.
    ascending:
        boolean. Specifies if the array is sorted in an ascending way or not. Default is True.

    Returns
    -------
    Index
    '''

    first = 0
    last = thresholds.shape[0] - 1

    # Binary search
    while first <= last:
        middle = (first + last) // 2
        
        if thresholds[middle] == thr:
            # Returns threshold's position
            return middle
        
        if ascending:
            if thresholds[middle] > thr:
                last = middle - 1
            else:
                first = middle + 1
        else:
            if thresholds[middle] < thr:
                last = middle - 1
            else:
                first = middle + 1

    # Returns position where 'thr' should be placed 
    return first


def get_fpr_tpr_at_threshold(fpr, tpr, thresholds, threshold):

    thr_pos = find_closest_value(thresholds, threshold, ascending=False)

    if thr_pos >= thresholds.shape[0]:
        thr_pos = thresholds.shape[0] - 1

    if thresholds[thr_pos] == threshold:
        return fpr[thr_pos], tpr[thr_pos]

    # Get fpr and tpr as an interpolated point

    threshold_dif = thresholds[thr_pos-1] - thresholds[thr_pos] # Remember: descending sorted thresholds array

    threshold_dif_thr = thresholds[thr_pos-1] - threshold 

    # Higher position in threshold array -> lower threshold value and higher values in fpr, tpr arrays

    fpr_dif = fpr[thr_pos] - fpr[thr_pos - 1]
    tpr_dif = tpr[thr_pos] - tpr[thr_pos - 1]

    dist_on_ROC = np.sqrt(fpr_dif**2 + tpr_dif**2)

    dist_on_ROC_thr = dist_on_ROC * (threshold_dif_thr / threshold_dif)

    # Applying Thales theorem
    # Interpolates fpr and tpr values
    fpr_dif_thr = dist_on_ROC_thr * (fpr_dif / dist_on_ROC) + fpr[thr_pos - 1]

    tpr_dif_thr = dist_on_ROC_thr * (tpr_dif / dist_on_ROC) + tpr[thr_pos - 1]

    return fpr_dif_thr, tpr_dif_thr


def find_sensibility_operative_point(fpr, tpr, thresholds, op_sens):

    sens_pos = find_closest_value(tpr, op_sens)

    if tpr[sens_pos] == op_sens:
        return fpr[sens_pos], tpr[sens_pos], thresholds[sens_pos]

    higher_tpr = tpr[sens_pos]
    lower_tpr = tpr[sens_pos - 1]

    proportion = (op_sens - lower_tpr) / (higher_tpr - lower_tpr)

    higher_fpr = fpr[sens_pos]
    lower_fpr = fpr[sens_pos - 1]

    interpolated_fpr = lower_fpr + proportion * (higher_fpr - lower_fpr)

    higher_thr = thresholds[sens_pos - 1]
    lower_thr = thresholds[sens_pos]

    interpolated_thr = get_interpolated_threshold(higher_fpr, 
                                                  lower_fpr, 
                                                  higher_tpr, 
                                                  lower_tpr, 
                                                  higher_thr, 
                                                  lower_thr, 
                                                  interpolated_fpr, 
                                                  op_sens)

    return interpolated_fpr, op_sens, interpolated_thr


def find_closest_higher_sensibility_operative_point(fpr, tpr, thresholds, op_sens):

    sens_pos = find_closest_value(tpr, op_sens)

    return fpr[sens_pos], tpr[sens_pos], thresholds[sens_pos]


def get_auc(fpr, tpr):

    '''
    Calculate an aproximated auc value by adding small rectangles and triangles.
    x axis: fpr
    y axis: tpr
    '''
    
    sum = 0

    for i, tpr_ in enumerate(tpr[:-1]):

        width = (fpr[i+1] - fpr[i])

        # Current height multiplied by current width
        sum += tpr_ * width

        # Small triangle at the top
        sum += (np.abs(tpr[i+1] - tpr_) * width) / 2

    return sum


def get_dr_levels_and_classifications_per_classes(ground_truth_and_dr, classes=None):

    '''
    This function picks and combines the columns specified by 'classes' from 'ground_truth'.

    Parameters
    -------
    ground_truth_and_dr
        True values for the current dataset/sample. It must be given in one-hot format, and it must
        have an auxiliar column which contains the real DR level for each element.
    classes
        It must be a list of integers or tuples, that will specify which columns have to be combined.
        Each column represents a label (for example, in 0vs1234, label 1 means DR levels 1,2,3 and 4)
        Default is None, in this case, all labels will be selected.

    Returns
    -------
    dr_levels_list
        List of arrays. Each array will contain all DR levels that constitute the current class.
    dr_classifications_list
        List of arrays. Each array will indicate which elements belong to the current class.
    '''

    if classes is None:
        classes = list(range(ground_truth_and_dr.shape[1] - 1))
    elif type(classes) != list:
        classes = [classes]

    dr_levels_list = []
    dr_classifications_list = []

    for i in classes:

        y_true_i = get_joined_array(ground_truth_and_dr, class_=i)

        # Get which DR levels constitute the selected class (this class could be multiple labels)
        dr_levels = np.sort(np.unique(ground_truth_and_dr[np.where(y_true_i == 1), -1]))

        # Get a classification array for each DR level vs others
        # This array is necessary to compute FNR for each DR level
        dr_classifications = tuple(np.where(ground_truth_and_dr[:,-1] == current_dr, 1, 0) for current_dr in dr_levels)

        dr_levels_list.append(dr_levels)
        dr_classifications_list.append(dr_classifications)

    return dr_levels_list, dr_classifications_list


# Calculate sensibility and specifity at operative sensibility value or threshold
def get_sensibility_and_fn_rates(fpr, 
                                 tpr, thresholds, 
                                 model_prediction, 
                                 dr_classifications, 
                                 operative_sens=None, 
                                 interpolate_sens_points=True,
                                 operative_th=None):

    if operative_sens is None:

        # Get fpr and tpr by interpolation
        fpr_, tpr_ = get_fpr_tpr_at_threshold(fpr, tpr, thresholds, operative_th)
        thr = operative_th

            # Get fpr and tpr using closest (existing) threshold
            # thr_index = find_closest_value(thresholds, operative_th, ascending=False)

            # thr_index = thr_index if np.abs(thresholds[thr_inx] - operative_th) < np.abs(thresholds[thr_inx - 1] - operative_th) else thr_index - 1

            # fpr_ = fpr[thr_index]
            # tpr_ = tpr[thr_index]
            # thr = thresholds[thr_index]
        
    else:
        if interpolate_sens_points:
            # Get fpr, tpr and threshold by interpolation
            # Find in ROC curve where is exactly located the point where sensibility is equal to 'operative_sens'
            fpr_, tpr_, thr = find_sensibility_operative_point(fpr, tpr, thresholds, operative_sens)
        else:
            # Find in ROC curve the first threshold whose tpr value is greater than given operative sensibility
            fpr_, tpr_, thr = find_closest_higher_sensibility_operative_point(fpr, tpr, thresholds, operative_sens)

    # False positive rate = 1 - true positive rate (aka. sensibility)
    fnr = 1 - tpr_

    fnr_dr_levels = []

    fn_per_dr_levels = []

    # Calculate false negative rate for each DR level in this sample, not in the infinite population
    for dr_class in dr_classifications:
        tp, fp, tn, fn = get_tp_fp_tn_fn(dr_class, model_prediction > thr)

        fnr_dr_levels.append(fn / (tp + fn))

        fn_per_dr_levels.append(fn)

    # Return 
    # Sensibility -> true positive rate
    # Specifity -> 1 - false positive rate
    # False negative rate (theoretical, value calculated)
    # False negative rates for each dr level in class '1' (in current sample of images)
    # Threshold
    # Number of false negatives for each dr level

    return tpr_, 1 - fpr_, fnr, fnr_dr_levels, thr, fn_per_dr_levels


def get_results_at_operative_points(false_positive_rates,

                                    true_positive_rates,
                                    thresholds,
                                    prediction,
                                    classes,
                                    dr_levels,
                                    dr_classifications,
                                    operative_sens_points=None, # Sensibility operative points
                                    interpolate_sens_points=True,
                                    operative_thresholds=None): # Operative thresholds obtanined from other dataset

    '''
    

    Parameters:
    - false positive rates (np.ndarray or list of ndarray): from ROC curve
    - true positive rates (np.ndarray or list of ndarray): from ROC curve
    - thresholds (np.ndarray or list of ndarray): used to build ROC curve
    - prediction (np.ndarray or list of ndarray): model's output
    - classes (list of None): how model classes should be grouped
    - dr_levels (np.ndarray or list of np.ndarray): sorted array of all dr levels present in class '1'
    - dr_classifications (list or None): array of len(dr_levels) arrays, each of them will have '1' only for a specific DR level
    - operative_sens_points (list or None): sensibility levels desired for each ROC
    - interpolate_sens_points (bool): specify if you want the exact point on ROC where operative_sens_point is, or get first real point whose sens is higher or equal
    - operative_thresholds (list of tuples or None): thresholds used on other dataset. Each tuple contains a threshold per ROC
    '''

    # Check that every argument (except operative points) are lists. If not, convert into 1-size lists
    if type(false_positive_rates) != list:
        false_positive_rates = [false_positive_rates]

    if type(true_positive_rates) != list:
        true_positive_rates = [true_positive_rates]

    if type(thresholds) != list:
        thresholds = [thresholds]

    if type(dr_levels) != list:
        dr_levels = [dr_levels]

    if type(dr_classifications) != list:
        dr_classifications = [dr_classifications]

    if prediction.ndim == 1:
        prediction = np.expand_dims(prediction, axis=1)

    results_for_each_ROC = []

    false_negatives_for_each_ROC = []

    for ROC_i in range(len(false_positive_rates)):

        # Get current ROC info
        false_positive_rates_i = false_positive_rates[ROC_i]
        true_positive_rates_i = true_positive_rates[ROC_i]
        thresholds_i = thresholds[ROC_i]

        prediction_i = get_joined_array(prediction, classes[ROC_i])
        # prediction_i = prediction[:, ROC_i]

        dr_levels_i = dr_levels[ROC_i]
        dr_classifications_i = dr_classifications[ROC_i]

        # Needed info
        sensibility = []
        specificity = []
        false_negative_rate = []
        false_negative_rate_dr_levels =  [[] for dr_lvl in dr_levels_i] # an empty list for each dr level in current ROC
        used_thresholds = []

        false_negatives_absolute = [[] for dr_lvl in dr_levels_i]

        if operative_sens_points is not None:

            # Dado un valor objetivo de sensibilidad, extraer mediante interpolación los valores de especificidad,
            # umbral y tasa de falsos negativos 

            for op in operative_sens_points:
                
                sens, sp, fn, fn_dr_lvls, thr, fns = get_sensibility_and_fn_rates(false_positive_rates_i, 
                                                                                  true_positive_rates_i, 
                                                                                  thresholds_i, 
                                                                                  prediction_i, 
                                                                                  dr_classifications_i, 
                                                                                  operative_sens=op/100,
                                                                                  interpolate_sens_points=interpolate_sens_points)

                sensibility.append(round(sens * 100, 2))
                specificity.append(round(sp * 100, 2))
                false_negative_rate.append(round(fn * 100, 2))
                
                for i, fn_dr in enumerate(fn_dr_lvls):
                    false_negative_rate_dr_levels[i].append(fn_dr)
                
                for i, fn_ in enumerate(fns):
                    false_negatives_absolute[i].append(fn_) 
                    
                used_thresholds.append(thr)
            
        if operative_thresholds is not None:

            # Dado un umbral de aplicación, obtener mediante interpolación (excepto las tasas de falsos negativos 
            # por cada clase, que se calculan con la muestra) los restantes datos del punto que corresponda en la
            # ROC

            for op_th in operative_thresholds:

                # Calculate results for given thresholds for current ROC
                sens, sp, fn, fn_dr_lvls, _, fns = get_sensibility_and_fn_rates(false_positive_rates_i, 
                                                                                true_positive_rates_i, 
                                                                                thresholds_i, 
                                                                                prediction_i, 
                                                                                dr_classifications_i, 
                                                                                operative_th=op_th[ROC_i])
                    
                sensibility.append(round(sens * 100, 2))
                specificity.append(round(sp * 100, 2))
                false_negative_rate.append(round(fn * 100, 2))

                for i, fn_dr in enumerate(fn_dr_lvls):
                    false_negative_rate_dr_levels[i].append(fn_dr)

                for i, fn_ in enumerate(fns):
                    false_negatives_absolute[i].append(fn_)              

                used_thresholds.append(op_th[ROC_i])

        # Save results    
        results = {}

        if operative_sens_points is not None:
            results['Sens. Operative point'] = operative_sens_points + ['Closest point']

        results['Sensibility'] = sensibility
        results['Specificity'] = specificity
        results['False negative rate'] = false_negative_rate

        # Add false negative rates for each DR level in class '1'
        # Tasa de Falsos Negativos de la muestra actual por cada clase especificada, 
        # no ha sido calculada a partir de la ROC (no es interpolación)
        for dr_lvl, fnr_dr in zip(dr_levels_i, false_negative_rate_dr_levels):
            results['FNR_' + str(dr_lvl) + ' (sample)'] = list(map(lambda x: round(x * 100, 2), fnr_dr))
            
        results['Thresholds'] = used_thresholds

        results_for_each_ROC.append(pd.DataFrame(results))


        false_negatives_for_each_ROC.append({dr_lvl: fn_ for dr_lvl, fn_ in zip(dr_levels_i, false_negatives_absolute)})

    return results_for_each_ROC, false_negatives_for_each_ROC


def get_predicted_labels(pred, thresholds):

    '''
    Compute the predicted labels.

    Parameters
    -------
    pred
        Model's prediction. Probabilities of belonging to each class/label.
    thresholds
        RD thresholds. There must be at least one threshold per non-healthy class. If the prediction is greater than one
        of these thresholds, the image will be marked as 'RD presence'.

    Returns
    -------
    pred_lb
        Predicted labels for each element in 'pred' array.
    '''

    if pred.shape[1] > 2:
        # There are multiple outputs and several thresholds (one per defined ROC)

        pred_lb = np.zeros((pred.shape[0],))
        
        for i in range(pred.shape[0]):
            
            # Se considera que hay un umbral por cada clase que signifique nivel RD 1 o mayor
            # Por ejemplo, si se establecen 3 salidas: 0 vs 1 vs 234, habría al menos un umbral para las clases 1:(1) y 2:(234)
            # Además, puede existir un último umbral extra que sea la probabilidad de padecer cualquier nivel de RD (mezclando
            # todas las clases que no sean estar sanos)

            # Se toma como umbral de si tiene RD el valor que tenga la salida 0, es decir, la probabilidad de estar sano
            # En caso de haber un umbral extra que mezcle todos los niveles de RD, se tomará como umbral el mínimo entre éste
            # y el valor de la salida 0
            has_DR_threshold = pred[i, 0] if (len(thresholds) == pred.shape[1] - 1) else np.minimum(pred[i, 0], thresholds[-1])
            
            # Si la suma de las probabilidades de pertenecer a cualquiera de las clases con RD igual o mayor a 1 supera este umbral,
            # se considerará que la imagen padece RD
            if np.sum(pred[i,1:]) > has_DR_threshold:
                values = np.copy(pred[i,1:])

                # Primero se busca si alguno de los valores de las siguientes columnas (clase 1, clase 2)
                # supera su correspondiente umbral. En dicho caso se asignará su clase como predicción
                for _ in range(len(values)):
                    max_ = np.argmax(values)
                    if values[max_] > thresholds[max_]:
                        pred_lb[i] = max_+1
                        break
                    # 'else' -- class with higher probability is still lower than its threshold
                    # so, others classes will be checked
                    values[max_] = -1
                
                # If no assignment was made although the sum of probabilities is higher than the probability of being healthy
                # Mark current image as the class that has the higher probability
                if pred_lb[i] == 0:
                    pred_lb[i] = np.argmax(pred[i,1:]) + 1

    else:
        # There's only one output -- Binary classification
        # Apply current threshold
        pred_lb = np.where(pred[:,1] > thresholds[0], 1, 0)
    
    return pred_lb


def get_accuracy(true, pred, thresholds, return_pred=False):
    
    assert len(thresholds) == pred.shape[1] - 1 or len(thresholds) == pred.shape[1] # En este caso, se proporciona un umbral para la suma de las
                                                                                    # clases enfermas

    # Get label with higher value. Remove last column because it is DR level
    true_lb = np.argmax(true[:,:-1], axis=1)

    pred_lb = get_predicted_labels(pred, thresholds)
    
    if not return_pred:
        return np.sum(true_lb == pred_lb) / true_lb.shape[0]
    else:
        return np.sum(true_lb == pred_lb) / true_lb.shape[0], pred_lb


def get_false_negatives_per_dr_level(true_lb, pred_lb):
    
    # DR levels except 0
    existing_fns_per_dr_lvl = {x: [] for x in np.sort(np.unique(true_lb[:,-1])) if x != 0}
    
    true_labels = np.argmax(true_lb[:,:-1], axis=1)
    
    missclassified_elements = np.where(true_labels != pred_lb)
    
    for x in existing_fns_per_dr_lvl:
        
        current_dr_lvl_images = np.where(true_lb[:,-1] == x)
        
        # Intersection between missclassified elements and those elements that belong to current dr level
        # These elements will be false negatives
        
        existing_fns_per_dr_lvl[x].extend(np.intersect1d(current_dr_lvl_images, missclassified_elements, assume_unique=True))
    
    return existing_fns_per_dr_lvl


def create_confusion_matrix(ground_truth, pred_lb, num_classes, DR_LEVELS_PER_CLASS):

    '''
    Computes confusion matrix

    Parameters
    -------
    ground_truth
        True values for the current dataset/sample. It must be given in one-hot format, and it must
        have an auxiliar column which contains the real DR level for each element.
    pred_lb
        Model's prediction for current dataset. This array must be given as an one-column array,
        specifying the predicted class for each element.
    num_classes
        Number of classes
    DR_LEVELS_PER_CLASS
        List of lists that specifies which DR levels belong to a label/class

    Returns
    -------
    matrix
        2D numpy array (num_classes x num_classes)
    dr_lvls_per_box
        List of dictionaries with counters for each matrix cell
    '''
    
    # True classes/labels (0 or 1 -- 0 vs 1234, 0 or 1 or 2 -- 0 vs 1 vs 234)
    true_lb = np.argmax(ground_truth[:,:-1], axis=1)
    
    true_dr_lvls = ground_truth[:,-1]
    
    matrix = np.zeros((num_classes, num_classes))
    
    # Cada casilla de la matriz de confusión llevará un contador de cuántas imágenes caen dentro
    # Este contrador se representa como un diccionario, permitiendo diferenciar entre todos los niveles
    # existentes de RD
    dr_lvls_per_box = [{dr:0 for dr in [item for sublist in DR_LEVELS_PER_CLASS for item in sublist]} for _ in range(num_classes**2)]
    
    for lb in range(num_classes):
        
        # Select all elements of current class/label
        current_true_lb = np.where(true_lb == lb)[0]
        
        for ind in current_true_lb:
            # Get prediction
            predicted_lb = int(pred_lb[ind])
            # Add to matrix
            matrix[lb][predicted_lb] += 1
            
            # Add to current cell counter
            dr_lvls_per_box[lb * num_classes + predicted_lb][int(true_dr_lvls[ind])] += 1
    
    # Iterate over all the cells and check their counter dictionaries
    for i in range(len(dr_lvls_per_box)):
        
        box = dr_lvls_per_box[i]
        
        # Apuntar los niveles de RD que no tienen imágenes en la celda actual
        to_delete = [k for k in box.keys() if box[k] == 0]
        
        # Create a copy without the empty values
        new_auxiliar_dict = {new_k: box[new_k] for new_k in box.keys() if new_k not in to_delete}
        
        dr_lvls_per_box[i] = new_auxiliar_dict
    
    return matrix, dr_lvls_per_box




