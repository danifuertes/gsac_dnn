import os
import sys
import cv2
import math
import json
import glob
import shutil
import operator
import argparse
import numpy as np
import matplotlib.pyplot as plt

from options import str2bool


def get_map_options():
    parser = argparse.ArgumentParser()

    # Model results
    parser.add_argument('--results_dir', type=str, default='model_date_test_date',
                        help="Folder with the results of test.py")
    parser.add_argument('--img_width', type=int, default=224, help="Target Image Width")
    parser.add_argument('--img_height', type=int, default=224, help="Target Image Height")

    # Bounding boxes or point-based detections
    parser.add_argument('--use_bb', type=str2bool, default=False,
                        help="True to use bounding boxes. False to use point-based detections")
    parser.add_argument('--min_overlap', type=float, default=0.5, help="PASCAL VOC2012 IoU threshold (use_bb=True)")
    parser.add_argument('--min_dist', type=float, default=16, help="Distance threshold (use_bb=False)")

    # Additional options
    parser.add_argument('--show_animation', type=str2bool, default=False, help="True to show animation")
    parser.add_argument('--draw_plot', type=str2bool, default=False, help="True to draw plot")
    parser.add_argument('--ignore', nargs='+', type=str, help="Ignore a list of classes")
    parser.add_argument('--set-class-iou', nargs='+', type=str,
                        help="Set IoU for a specific class (e.g., python map.py --set-class-iou person 0.7)")
    opts = parser.parse_args()

    '''
        0,0 ------> x (width)
         |
         |  (Left,Top)
         |      *_________
         |      |         |
                |         |
         y      |_________|
      (height)            *
                    (Right,Bottom)
    '''
    opts.img_size = (opts.img_width, opts.img_height)
    return opts


def error(msg):
    """
     throw error and exit
    """
    print(msg)
    sys.exit(0)


def is_float_between_0_and_1(value):
    """
     check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # If there were no detections of the current class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    # False Positives Per Image
    fppi = fp_cumsum / float(num_images)
    fppi_tmp = np.insert(fppi, 0, -1.0)

    # Miss Rate
    mr = (1 - precision)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


def voc_ap(rec, prec):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """

    # Insert 0 at the beginning and end of rec and prec
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    # Make the precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    # Create a list of indexes where the recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    # Calculate the Average Precision (AP) as the area under the curve
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    """
     Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    """
     Draws text in image
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    line_type = 1
    bottom_left_corner_of_text = pos
    cv2.putText(img, text, bottom_left_corner_of_text, font, font_scale, color, line_type)
    text_width, _ = cv2.getTextSize(text, font, font_scale, line_type)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # Get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi

    # Get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width

    # Get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    """
     Draw plot using Matplotlib
    """

    # Sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))

    # Unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)

    """
     Special case to draw in:
        - green -> TP: True Positives (object detected and matches ground-truth)
        - red -> FP: False Positives (object detected but does not match ground-truth)
        - orange -> FN: False Negatives (object not detected but present in the ground-truth)
    """
    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        plt.legend(loc='lower right')

        # Write number on side of bar
        fig, axes = plt.gcf(), plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)

            # Trick to paint multicolor with offset: first paint everything and then re-paint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(r, t, fig, axes)

    # Write number on side of bar
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig, axes = plt.gcf(), plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')

            # Re-set axes to show number inside the figure
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)

    # Set window title
    fig.canvas.set_window_title(window_title)

    # Write classes in y-axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)

    # Re-scale height accordingly
    init_height = fig.get_figheight()

    # Compute the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi

    # Compute the required figure height
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)

    # Set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # Set plot title
    plt.title(plot_title, fontsize=14)

    # Set axis titles
    plt.xlabel(x_label, fontsize='large')

    # Adjust size of window
    fig.tight_layout()

    # Save the plot
    fig.savefig(output_path)

    # Show image
    if to_show:
        plt.show()

    # Close the plot
    plt.close()


def main(opts):

    # If there are no classes to ignore then replace None by empty list
    opts.ignore = [] if opts.ignore is None else opts.ignore

    # Check if there are classes with specific IoU
    specific_iou_flagged = False if opts.set_class_iou is None else True

    # Make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Directories where detections and ground-truth files are saved after running test.py
    GT_PATH = os.path.join(os.getcwd(), 'predictions', opts.results_dir, 'ground-truth')
    DR_PATH = os.path.join(os.getcwd(), 'predictions', opts.results_dir, 'detection-results')

    # If there are no images then no animation can be shown
    IMG_PATH = os.path.join(os.getcwd(), 'predictions', opts.results_dir, 'images')
    if os.path.exists(IMG_PATH):
        for _, _, files in os.walk(IMG_PATH):
            # No image files found
            if not files:
                opts.show_animation = False
    else:
        opts.show_animation = False

    # Create a directory for the temporal files
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    # Create the results directory
    results_files_path = os.path.join(os.getcwd(), 'predictions', opts.results_dir, 'results')
    if not os.path.exists(results_files_path):
        os.makedirs(results_files_path)

    # Create dirs for plot/animation
    if opts.draw_plot:
        os.makedirs(os.path.join(results_files_path, "classes"))
    if opts.show_animation:
        os.makedirs(os.path.join(results_files_path, "images", "detections_one_by_one"))

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # Get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    # For each group of annotations in every image
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # Check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error("Error. File not found: {}\n".format(temp_path))
        lines_list = file_lines_to_list(txt_file)

        # For each annotation in an image
        bounding_boxes, already_seen_classes, is_difficult = [], [], False
        for line in lines_list:

            # Load annotation data
            try:
                if opts.use_bb:
                    if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, left, top, right, bottom = line.split()
                else:
                    if "difficult" in line:
                        class_name, x, y, _difficult = line.split()
                        is_difficult = True
                    else:
                        class_name, x, y = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error(error_msg)

            # Check if class is in ignore list, if yes skip
            if class_name in opts.ignore:
                continue

            # Save annotation
            bbox = left + " " + top + " " + right + " " + bottom if opts.use_bb else x + " " + y
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # If class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # If class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # Dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    # Sort the classes alphabetically and count them
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
     Check format of the flag --set-class-iou (if used) -> [class_1] [IoU_1] [class_2] [IoU_2]
    """
    if specific_iou_flagged:
        n_args = len(opts.set_class_iou)
        error_msg = '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'

        # Number of arguments has to be an even number
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)

        # Classes
        specific_iou_classes = opts.set_class_iou[::2]

        # Values
        iou_list = opts.set_class_iou[1::2]

        # Check if there are the same number of classes and values
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)

        # For each class, check if class is in the ground-truth
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)

        # For each value, check if value is in the range [0, 1]
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # Get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    # For each class
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []

        # For each group of detections in every image
        for txt_file in dr_files_list:

            # Use file_id to associate detections with ground-truth of the same image
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))

            # The first time, check if all the corresponding ground-truth files exist
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error("Error. File not found: {}\n".format(temp_path))

            # For each detection in an image
            lines = file_lines_to_list(txt_file)
            for line in lines:

                # Load detection data
                try:
                    if opts.use_bb:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    else:
                        tmp_class_name, confidence, x, y = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)

                # Save detection
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom if opts.use_bb else x + " " + y
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})

        # Sort detection-results by decreasing confidence and dump them into a ".json" file
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_ap, sum_fscore, sum_lamr, sum_mr, sum_fppi = 0.0, 0.0, 0, 0, 0
    ap_dictionary, lamr_dictionary = {}, {}

    # Open files to store the results
    results_file_ap = open(results_files_path + "/results.txt", 'w')
    results_file_fscore = open(results_files_path + "/results_fscore.txt", 'w')
    results_file_lamr = open(results_files_path + "/results_lamr.txt", 'w')
    results_file_ap.write("# AP and precision/recall per class\n")
    results_file_fscore.write("# F-score and best precision/recall per class\n")
    results_file_lamr.write("# Log Average Miss Rate per class\n")

    # For each class
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0

        # Load detection-results of that class
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        # Assign detection-results to ground-truth objects
        tp = [0] * len(dr_data)
        fp = [0] * len(dr_data)
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]

            # Show animation
            if opts.show_animation:

                # Find ground truth image
                ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                if len(ground_truth_img) == 0:
                    error("Error. Image not found with id: " + file_id)
                elif len(ground_truth_img) > 1:
                    error("Error. Multiple images with id: " + file_id)

                # Image found
                else:

                    # Load image
                    img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])

                    # Load image with draws of multiple detections
                    img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
                    if os.path.isfile(img_cumulative_path):
                        img_cumulative = cv2.imread(img_cumulative_path)
                    else:
                        img_cumulative = img.copy()

                    # Add bottom border to image
                    bottom_border = 60
                    black = [0, 0, 0]
                    img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=black)

            # Assign detection-results to ground truth object if any open ground-truth with that file_id
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1 if opts.use_bb else min(opts.img_size)
            gt_match = -1

            # Load detected object
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:

                # Look for a class_name match
                if obj["class_name"] == class_name:

                    # Load annotation
                    bbgt = [float(x) for x in obj["bbox"].split()]

                    # Compute IoU metric for point-based detections
                    if opts.use_bb:
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1

                        # Compute IoU = area of intersection (iw * ih) / area of union (ua)
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                 (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                    # Compute euclidean distance metric for point-based detections
                    else:
                        ov = math.sqrt((bb[0] - bbgt[0]) ** 2 + (bb[1] - bbgt[1]) ** 2)
                        if ov < ovmax:
                            ovmax = ov
                            gt_match = obj

            # Assign detection as true positive/don't care/false positive
            if opts.show_animation:
                status = "NO MATCH FOUND!"  # status is only used in the animation

            # Set minimum overlap
            min_overlap = opts.min_overlap if opts.use_bb else opts.min_dist
            if specific_iou_flagged:
                if class_name in specific_iou_classes:
                    index = specific_iou_classes.index(class_name)
                    min_overlap = float(iou_list[index])

            # Bounding boxes: IoU >= IoU threshold | Point-based detections: Distance <= Distance threshold
            condition = ovmax >= min_overlap if opts.use_bb else ovmax <= min_overlap
            if condition:
                if "difficult" not in gt_match:

                    # True positive
                    if not bool(gt_match["used"]):
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1

                        # Update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                        if opts.show_animation:
                            status = "MATCH!"

                    # False positive (multiple detection)
                    else:
                        fp[idx] = 1
                        if opts.show_animation:
                            status = "REPEATED MATCH!"

            # False positive
            else:
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

            """
             Draw image to show animation
            """
            if opts.show_animation:
                height, _ = img.shape[:2]

                # Radius for the points of single-point detection
                radius = 10

                # Colors (OpenCV works with BGR)
                white = (255, 255, 255)
                light_blue = (255, 200, 100)
                green = (0, 255, 0)
                light_red = (30, 30, 255)

                # 1st line
                margin = 10
                v_pos = int(height - margin - (bottom_border / 2.0))
                text = "Image: " + ground_truth_img[0] + " "
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                if ovmax != -1:
                    color = light_red
                    if status == "INSUFFICIENT OVERLAP":
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                    else:
                        text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                        color = green
                    img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                # 2nd line
                v_pos += int(bottom_border / 2.0)
                rank_pos = str(idx + 1)
                text = "Detection #rank: " + rank_pos + \
                       " confidence: {0:.2f}% ".format(float(detection["confidence"]) * 100)
                img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                color = green if status == "MATCH!" else light_red
                text = "Result: " + status + " "
                img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                font = cv2.FONT_HERSHEY_SIMPLEX

                bb = [int(i) for i in bb]
                if opts.use_bb:

                    # If there are intersections between the bounding-boxes
                    if ovmax > 0:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                else:
                    if ovmax > 0:  # if there is intersections between the bounding-boxes
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.circle(img, (bbgt[0], bbgt[1]), radius, light_blue, 2)
                        cv2.circle(img_cumulative, (bbgt[0], bbgt[1]), radius, light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    cv2.circle(img, (bb[0], bb[1]), radius, color, 2)
                    cv2.circle(img_cumulative, (bb[0], bb[1]), radius, color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)

                # Show image for 20 ms
                cv2.imshow("Animation", img)
                cv2.waitKey(20)

                # Save image to results
                output_img_path = results_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + \
                                  str(idx) + ".jpg"
                cv2.imwrite(output_img_path, img)

                # Save the image with all the objects drawn to it
                cv2.imwrite(img_cumulative_path, img_cumulative)

        # Count false positives
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        # Count true positives
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        # Compute Recall
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        round_rec = ['%.2f' % elem for elem in rec]

        # Compute Precision
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        round_prec = ['%.2f' % elem for elem in prec]

        # Compute AP
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_ap += ap
        text_ap = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
        results_file_ap.write(text_ap + "\n Precision: " + str(round_prec) + "\n Recall :" + str(round_rec) + "\n\n")
        print(text_ap)
        ap_dictionary[class_name] = ap

        # Compute F1-Score
        eps = np.finfo(float).eps
        fscore = [2 * (p * r) / (p + r + eps) for p, r in zip(prec, rec)]
        best_fscore = 2 * (prec[-1] * rec[-1]) / (prec[-1] + rec[-1] + eps) if len(fscore) > 0 else 0
        round_fscore = ['%.2f' % elem for elem in [best_fscore]]
        sum_fscore += best_fscore
        text_fscore = "{0:.2f}%".format(best_fscore * 100) + " = " + class_name + " F-score "
        results_file_fscore.write(text_fscore + '\n f1-score: %s' % str(round_fscore))
        results_file_fscore.write('\n Precision: %s' % prec[-1])
        results_file_fscore.write('\n Recall: %s' % rec[-1] + "\n\n")
        print(text_fscore)

        # Compute LAMR
        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        sum_lamr += lamr
        text4 = "{0:.2f}%".format(lamr * 100) + " = " + class_name + " lamr "
        lamr_dictionary[class_name] = lamr

        # Compute MR and FPPI
        rounded_mr = ['%.2f' % elem for elem in mr]
        rounded_fppi = ['%.2f' % elem for elem in fppi]
        sum_mr += mr[-1]
        sum_fppi += fppi[-1]
        results_file_lamr.write(text4 + "\n mr: " + str(rounded_mr) + "\n fppi :" + str(rounded_fppi))
        results_file_lamr.write('\n MR: %s' % (rounded_mr[-1]))
        results_file_lamr.write('\n FPPI: %s' % (rounded_fppi[-1]) + "\n\n")

        """
         Draw plot
        """
        if opts.draw_plot:

            # Plot Precision-Recall curve
            plt.plot(rec, prec, '-o')

            # add a new penultimate point to the list (mrec[-2], 0.0) since the last line segment (and respective area)
            # do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

            # Set window title
            fig = plt.gcf()
            fig.canvas.set_window_title('AP ' + class_name)

            # Set plot title
            plt.title("Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " ")

            # Set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')

            # Set axes
            axes = plt.gca()
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])

            # Save the plot and clear axes for next plot
            fig.savefig(results_files_path + "/classes/" + class_name + ".png")
            plt.cla()

            # Plot FPPI-MR curve
            plt.plot(fppi, mr)

            # Set window title
            fig = plt.gcf()
            fig.canvas.set_window_title('LAMR ' + class_name)

            # Set axis titles
            plt.title('class: {:.2f}% '.format(lamr * 100) + class_name + ' LAMR')

            # Set axis titles
            plt.xlabel('FPPI')
            plt.ylabel('MR')

            # Set log scale in x-axis
            plt.xscale('log')

            # Set axes
            axes = plt.gca()
            min_fppi = np.min(fppi) if np.min(fppi) > 0 else 1e-5
            max_fppi = np.max(fppi) if np.max(fppi) > 0 else 1
            axes.set_xlim([10 ** math.floor(math.log10(min_fppi)), 10 ** math.ceil(math.log10(max_fppi))])

            # Save the plot and clear axes for next plot
            fig.savefig(results_files_path + "/classes/" + class_name + "_lamr.png")
            plt.cla()

    # Close windows
    if opts.show_animation:
        cv2.destroyAllWindows()

    # Print / Write results
    results_file_ap.write("\n# mAP of all classes\n")
    print('\nAverage over classes:')

    # Print / Write mAP
    mean_ap = sum_ap / n_classes
    text_ap = "mAP = {0:.2f}%".format(mean_ap * 100)
    results_file_ap.write(text_ap + "\n")
    print(text_ap)

    # Print / Write F1-Score
    mean_fscore = sum_fscore / n_classes
    text_fscore = "fscore = {0:.2f}%".format(mean_fscore * 100)
    results_file_fscore.write(text_fscore + "\n")
    print(text_fscore)

    # Print / Write LAMR
    mean_lamr = sum_lamr / n_classes
    text_lamr = "lamr = {0:.2f}%".format(mean_lamr * 100)
    results_file_lamr.write(text_lamr + "\n")
    print(text_lamr)

    # Print / Write MR
    mean_mr = sum_mr / n_classes
    text_mr = "mr = {0:.2f}%".format(mean_mr * 100)
    results_file_lamr.write(text_mr + "\n")
    print(text_mr)

    # Print / Write FPPI
    mean_fppi = sum_fppi / n_classes
    text_fppi = "fppi = {0:.2f}%".format(mean_fppi * 100)
    results_file_lamr.write(text_fppi + "\n")
    print(text_fppi)

    # Close files
    results_file_ap.close()
    results_file_fscore.close()
    results_file_lamr.close()

    # Remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    # Count total of detection-results
    det_counter_per_class = {}
    for txt_file in dr_files_list:

        # Get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]

            # Check if class is in ignore list, if yes skip
            if class_name in opts.ignore:
                continue

            # Count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1

            # If class didn't exist yet
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if opts.draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title, x_label, output_path, to_show,
                       plot_color, '',)

    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
     Finish counting true positives
    """
    # If class exists in detection-result but not in ground-truth then there are no true positives in that class
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    """
     Plot the total number of occurences of each class in the "detection-results" folder
    """
    if opts.draw_plot:
        window_title = "detection-results-info"
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(det_counter_per_class, len(det_counter_per_class), window_title, plot_title, x_label,
                       output_path, to_show, plot_color, true_p_bar)

    """
     Write number of detected objects per class to results.txt
    """
    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    """
     Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if opts.draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = results_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(lamr_dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       "")

    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if opts.draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mean_ap * 100)
        x_label = "Average Precision"
        output_path = results_files_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(ap_dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       "")


if __name__ == "__main__":
    main(get_map_options())
