import os
from .env import get_max_val, get_min_val
import math
import numpy as np
from matplotlib import pyplot as plt

config = {
    "text_size": {"pie": 16, "title": 18, "barh": 16}
}


def pie_plot(values, labels, title, save_path, save, colors=None):
    plt.clf()
    if colors is None:
        plt.pie(values, labels=labels, shadow=True, autopct='%1.1f%%',
                textprops={'fontsize': config["text_size"]["pie"]}, startangle=90)
    else:
        plt.pie(values, labels=labels, colors=colors, shadow=True, autopct='%1.1f%%',
                textprops={'fontsize': config["text_size"]["pie"]}, startangle=90)
    plt.title(title, fontsize=config["text_size"]["title"])
    if save:
        plt.savefig(save_path)
    plt.show()


def plot_class_distribution_of_fp(dict_to_plot, output_path, save):
    plt.clf()
    labels = [k[:2].upper() for k in dict_to_plot.keys()]
    values = [dict_to_plot[d] / sum(dict_to_plot.values()) for d in dict_to_plot.keys()]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    title = "False Positive distribution among categories"
    path_to_save = os.path.join(output_path, 'false_positive_cat_distribution.png')
    pie_plot(values, labels, title, path_to_save, save, colors=colors)


def plot_false_positive_errors(error_values, error_names, category_metric_value, category_name, output_path, save):
    labels = [label[:1].upper() for label in error_names]
    total_errors = [err[1] for err in error_values]
    percentage_error = [e / sum(total_errors) * 100 for e in total_errors]

    colors = ['lightskyblue', 'lightyellow', 'orchid', 'coral']
    title = "False Positive distribution of category {}".format(category_name)
    path_to_save = os.path.join(output_path, 'false_positive_category_{}-distribution.png'.format(category_name))
    pie_plot(percentage_error, labels, title, path_to_save, save, colors)

    # Bar Plot
    plt.clf()
    title = "False Positive impact for category {}".format(category_name)
    performance_values = [(err[0] - category_metric_value) for err in error_values]
    y_pos = np.arange(len(error_names))
    plt.barh(y_pos, performance_values, align='center', color=colors)
    plt.tick_params(labelsize=config["text_size"]["barh"])
    plt.yticks(y_pos, labels)
    plt.title(title, fontsize=config["text_size"]["title"])
    path_to_save = os.path.join(output_path, 'false_positive_category_{}-gain.png'.format(category_name))
    if save:
        plt.savefig(path_to_save)
    plt.show()


def plot_ap_bars(results, x, drawline):
    y_todraw = [0 if math.isnan(results[key]["value"]) else results[key]["value"] for key in results]
    y_error = []
    for key in results.keys():
        value = results[key]["std"]
        if value is None or math.isnan(value):
            y_error.append(0)
        else:
            y_error.append(value)

    plt.errorbar(x=x, y=y_todraw, yerr=y_error, ecolor='red', ls='none', lw=1, marker='_', markersize=8,
                 capsize=2)  # , marker='s', ,mec='green', ms=20, mew=4)

    if drawline:  # join all the points
        plt.plot(x, y_todraw, color='cornflowerblue')
    else:
        for i in range(0, len(x), 2):  # join every two points
            plt.plot(x[i: i + 2], y_todraw[i: i + 2], color='cornflowerblue')

    for k in range(len(y_todraw)):  # set the numbers text in bar errors
        plt.text(x[k], y_todraw[k], s='{:.2f}'.format(y_todraw[k]), fontsize=12)


def make_multi_category_plot(results, property_name, property_values_names, title_str, save_plot, plot_path):

    xtickstep = 1
    fs = 15
    drawline = True
    plt.clf()
    plt.style.use('seaborn-whitegrid')

    nobj = len(results.keys())
    rangex = np.zeros(1)
    maxy = 1
    xticks = np.asarray([])
    firsttick = np.zeros(nobj)
    for index, key in enumerate(results):
        if not property_name in results[key].keys():
            continue
        result = results[key][property_name]
        nres = len(result)
        space = 0 if index == 0 else 1
        rangex = rangex[-1] + space + np.arange(1, nres + 1)  # expand range x in plot according to the needed lines
        plot_ap_bars(result, rangex, drawline)
        ap = np.asarray([result[key]["value"] for key in result])

        maxy = max(maxy, np.around((np.amax(ap) + 0.15) * 10) / 10)
        ap_cat = results[key]['all']["value"]
        plt.plot([rangex[0], rangex[-1]], np.asarray([1, 1]) * ap_cat, linestyle='dashed', lw=1.5, color='black')
        firsttick[index] = rangex[0]
        xticks = np.concatenate((xticks, rangex[0:rangex.size:xtickstep]))

    maxy = min(maxy, 1)
    axes = plt.gca()
    axes.set_ylim([0, maxy])
    axes.set_xlim([0, rangex[-1] + 1])

    if len(property_values_names) == nres:  # TODO: CHECK THIS
        property_values_names = np.tile(property_values_names, nobj)
    plt.xticks(xticks, property_values_names)

    for index, key in enumerate(results):
        value = key
        if len(key) > 15:
            value = key[:2]
        if key != 'all':
            plt.text(firsttick[index], maxy + 0.02, value, fontsize=fs)

    plt.title(title_str, fontdict={'horizontalalignment': 'center',
                                   'fontsize': fs}, loc='center', pad=30)
    # plt.tight_layout()

    if save_plot:
        plot_path = os.path.join(plot_path, title_str + ".png")
        plt.savefig(plot_path, dpi=300)


def display_sensitivity_impact_plot(results, output_path, property_names, display_names, save):
    valid = [True] * len(property_names)
    maxp = np.zeros((len(results.keys()), len(property_names)))
    minp = np.zeros((len(results.keys()), len(property_names)))

    # For each category, obtain the max and min of each property the different possible values
    for c, category in enumerate(results.keys()):
        for p, prop in enumerate(property_names):
            if prop in results[category].keys():
                maxp[c, p] = get_max_val(results[category][prop], "value", None)  # Max for this category
                minp[c, p] = get_min_val(results[category][prop], "value", None)  # Min for this category
            else:
                valid[p] = False

    # Obtain mean of max/min values among the different categories for each property for the valid categories
    maxval = np.mean(maxp[:, valid], 0)
    minval = np.mean(minp[:, valid], 0)

    property_names = np.array(display_names)[valid]

    # Average of  AP among categories (do not consider the different properties)
    avgval = np.mean([results[cat]["all"]["value"] for cat in results.keys()])

    ##display graph
    plt.clf()
    plt.style.use('seaborn-whitegrid')
    fs = 12

    # Labels to show
    xticklab = np.asarray(property_names)
    xticklab = xticklab[valid]

    plt.rcParams['figure.figsize'] = (np.sum(valid) + 2, 4.5)  # dynamic width adaptation
    plt.plot([1, len(property_names)], [avgval, avgval], linestyle='dashed', lw=1,
             color='black')
    asymetric_error = [(avgval - minval), (maxval - avgval)]
    plt.errorbar(x=np.arange(1, len(property_names) + 1), y=(avgval * np.ones(len(property_names))),
                 yerr=asymetric_error,
                 ecolor='red',
                 ls='none', lw=1, marker='_', markersize=8, capsize=4, color='red')

    for x in range(len(property_names)):
        plt.text(x + 1.05, minval[x], s='{:.3f}'.format(minval[x]), fontsize=fs - 2)
        plt.text(x + 1.05, maxval[x], s='{:.3f}'.format(maxval[x]), fontsize=fs - 2)
    plt.text(0.6, avgval, s='{:.3f}'.format(avgval), fontsize=fs)

    plt.xticks(np.arange(1, len(property_names) + 1), xticklab)

    axis_x_val = plt.gca().get_xlim()
    plt.gca().set_xlim([0.55, axis_x_val[1]])

    plt.title('Sensitivity and Impact',
              fontdict={'horizontalalignment': 'center',
                        'fontsize': fs}, loc='center', pad=15)
    path_to_save = os.path.join(output_path, 'plots_impact_strong.png')

    if save:
        plt.savefig(path_to_save, dpi=300)
    plt.show()
