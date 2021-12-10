import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def eval(prediction_json, guided_prediction_json, save_path):
    with open(prediction_json) as f:
      predictions = json.load(f)
    with open(guided_prediction_json) as f:
      guided_predictions = json.load(f)

    # Compare single dice scores
    prediction_dices, guided_prediction_dices, diffs = [], [], []
    for i in range(len(predictions["results"]["all"])):
        prediction_dice = predictions["results"]["all"][i]["1"]["Dice"]
        guided_prediction_dice = guided_predictions["results"]["all"][i]["1"]["Dice"]
        diff = guided_prediction_dice - prediction_dice
        prediction_dices.append(prediction_dice)
        guided_prediction_dices.append(guided_prediction_dice)
        diffs.append(diff)
        print("prediction_dice: {}, guided_prediction_dice: {}, diff: {}".format(prediction_dice, guided_prediction_dice, diff))
    print("prediction_dice mean: {}, prediction_dice median: {}".format(np.mean(prediction_dices), np.median(prediction_dices)))
    print("guided_prediction_dice mean: {}, guided_prediction_dice median: {}".format(np.mean(guided_prediction_dices), np.median(guided_prediction_dices)))
    print("diff mean: {}, diff median: {}".format(np.mean(guided_prediction_dices) - np.mean(prediction_dices), np.median(guided_prediction_dices) - np.median(prediction_dices)))

    sns.set_theme(style="whitegrid")
    # sns.violinplot(x=prediction_dices)
    # sns.swarmplot(x=prediction_dices, color="k", alpha=0.8)

    # tips = sns.load_dataset("tips")
    # print(tips)
    # ax = sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted", split=True)

    # data = {"x": np.zeros(len(prediction_dices) + len(guided_prediction_dices)), "dices": np.concatenate([prediction_dices, guided_prediction_dices]), "guided": np.concatenate([np.resize(["Prediction"], len(prediction_dices)), np.resize(["Guided Prediction"], len(guided_prediction_dices))])}
    # sns.violinplot(x="x", y="dices", hue="guided", data=data, palette="muted", split=True)
    # sns.swarmplot(x="x", y="dices", hue="guided", data=data, split=True, color="k", alpha=0.8)

    data = {"x": np.zeros(len(guided_prediction_dices) + len(prediction_dices)), "dices": np.concatenate([guided_prediction_dices, prediction_dices]), "guided": np.concatenate([np.resize(["Guided Prediction"], len(guided_prediction_dices)), np.resize(["Prediction"], len(prediction_dices))])}
    violinplot = sns.violinplot(y="x", x="dices", hue="guided", data=data, palette="muted", split=True, orient="h")
    swarmplot = sns.swarmplot(y="x", x="dices", hue="guided", data=data, split=True, color="k", alpha=0.8, orient="h")
    swarmplot.set_yticks([])
    handles, labels = swarmplot.get_legend_handles_labels()
    legend = plt.legend(handles[0:2], labels[0:2], loc='upper left')
    plt.xlabel("Dice score")
    plt.ylabel("Number of cases")
    plt.savefig(save_path + "eval_mask_recommendation.png")

if __name__ == '__main__':
    task = "Task070_guided_all_public_ggo"
    prediction_json = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/basic_predictions_summary.json"
    guided_prediction_json = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/refined_predictions_summary.json"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/"

    eval(prediction_json, guided_prediction_json, save_path)