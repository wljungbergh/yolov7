from collections import defaultdict
import os
import json
import numpy as np

ROOT = "/workspaces/s0001387/repos/yolov7/runs/test"


MODES = ["blur", "dnat", "original"]
CLASSES = ["Vehicle", "Pedestrian", "VulnerableVehicle", "All"]
N_ITERATIONS = 3
NAME = "pretrained"

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR",
    "AR50",
    "AR75",
    "ARs",
    "ARm",
    "ARl",
]

TABLE = """
<table class="wrapped">
  <thead>
    <tr>
      <th align="center">Experiment</th>
      <th align="center">AP</th>
      <th align="center">AP50</th>
      <th align="center">AP75</th>
      <th align="center">APs</th>
      <th align="center">APm</th>
      <th align="center">APl</th>
      <th align="center">Vehicle AP</th>
      <th align="center">VulnerableVehicle AP</th>
      <th align="center">Pedestrian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Blur</td>
      <td align="center">30.168 ± 0.064</td>
      <td align="center">54.658 ± 0.096</td>
      <td align="center">28.826 ± 0.079</td>
      <td align="center">7.240 ± 0.019</td>
      <td align="center">30.499 ± 0.097</td>
      <td align="center">51.078 ± 0.100</td>
      <td align="center">42.407 ± 0.037</td>
      <td align="center">25.875 ± 0.126</td>
      <td align="center">22.223 ± 0.040</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">30.276 ± 0.032</td>
      <td align="center">54.857 ± 0.090</td>
      <td align="center">28.816 ± 0.077</td>
      <td align="center">7.152 ± 0.134</td>
      <td align="center">30.572 ± 0.063</td>
      <td align="center">51.308 ± 0.079</td>
      <td align="center">42.481 ± 0.020</td>
      <td align="center">25.905 ± 0.142</td>
      <td align="center">22.443 ± 0.025</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">30.231 ± 0.086</td>
      <td align="center">54.793 ± 0.056</td>
      <td align="center">28.723 ± 0.146</td>
      <td align="center">7.231 ± 0.041</td>
      <td align="center">30.488 ± 0.137</td>
      <td align="center">51.232 ± 0.071</td>
      <td align="center">42.408 ± 0.065</td>
      <td align="center">25.963 ± 0.155</td>
      <td align="center">22.323 ± 0.039</td>
    </tr>
  </tbody>
</table>
<h2>COCO results - main</h2>
<table class="wrapped">
  <thead>
    <tr>
      <th align="center">Train</th>
      <th align="center">Eval</th>
      <th align="center">AP</th>
      <th align="center">AP50</th>
      <th align="center">AP75</th>
      <th align="center">APs</th>
      <th align="center">APm</th>
      <th align="center">APl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Blur</td>
      <td align="center">Blur</td>
      <td align="center">30.116</td>
      <td align="center">54.562</td>
      <td align="center">28.693</td>
      <td align="center">7.273</td>
      <td align="center">30.479</td>
      <td align="center">50.981</td>
    </tr>
    <tr>
      <td align="center">Blur</td>
      <td align="center">DNAT</td>
      <td align="center">30.131</td>
      <td align="center">54.554</td>
      <td align="center">28.733</td>
      <td align="center">7.264</td>
      <td align="center">30.476</td>
      <td align="center">50.998</td>
    </tr>
    <tr>
      <td align="center">Blur</td>
      <td align="center">Original</td>
      <td align="center">30.117</td>
      <td align="center">54.558</td>
      <td align="center">28.714</td>
      <td align="center">7.266</td>
      <td align="center">30.476</td>
      <td align="center">50.975</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">Blur</td>
      <td align="center">30.277</td>
      <td align="center">54.988</td>
      <td align="center">28.870</td>
      <td align="center">7.210</td>
      <td align="center">30.645</td>
      <td align="center">51.126</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">DNAT</td>
      <td align="center">30.305</td>
      <td align="center">54.927</td>
      <td align="center">28.912</td>
      <td align="center">7.200</td>
      <td align="center">30.662</td>
      <td align="center">51.255</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">Original</td>
      <td align="center">30.308</td>
      <td align="center">54.931</td>
      <td align="center">28.918</td>
      <td align="center">7.208</td>
      <td align="center">30.657</td>
      <td align="center">51.281</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">Blur</td>
      <td align="center">30.317</td>
      <td align="center">54.746</td>
      <td align="center">28.897</td>
      <td align="center">7.288</td>
      <td align="center">30.649</td>
      <td align="center">51.142</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">DNAT</td>
      <td align="center">30.352</td>
      <td align="center">54.860</td>
      <td align="center">28.933</td>
      <td align="center">7.295</td>
      <td align="center">30.676</td>
      <td align="center">51.323</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">Original</td>
      <td align="center">30.352</td>
      <td align="center">54.863</td>
      <td align="center">28.921</td>
      <td align="center">7.289</td>
      <td align="center">30.661</td>
      <td align="center">51.329</td>
    </tr>
  </tbody>
</table>
<h2>COCO results - per category</h2>
<table class="wrapped">
  <thead>
    <tr>
      <th align="center">Train</th>
      <th align="center">Eval</th>
      <th align="center">Vehicle AP</th>
      <th align="center">VulnerableVehile AP</th>
      <th align="center">Pedestrian AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Blur</td>
      <td align="center">Blur</td>
      <td align="center">42.393</td>
      <td align="center">25.753</td>
      <td align="center">22.203</td>
    </tr>
    <tr>
      <td align="center">Blur</td>
      <td align="center">DNAT</td>
      <td align="center">42.389</td>
      <td align="center">25.748</td>
      <td align="center">22.257</td>
    </tr>
    <tr>
      <td align="center">Blur</td>
      <td align="center">Original</td>
      <td align="center">42.398</td>
      <td align="center">25.738</td>
      <td align="center">22.217</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">Blur</td>
      <td align="center">42.452</td>
      <td align="center">26.028</td>
      <td align="center">22.350</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">DNAT</td>
      <td align="center">42.458</td>
      <td align="center">26.050</td>
      <td align="center">22.406</td>
    </tr>
    <tr>
      <td align="center">DNAT</td>
      <td align="center">Original</td>
      <td align="center">42.463</td>
      <td align="center">26.046</td>
      <td align="center">22.416</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">Blur</td>
      <td align="center">42.479</td>
      <td align="center">26.186</td>
      <td align="center">22.285</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">DNAT</td>
      <td align="center">42.499</td>
      <td align="center">26.186</td>
      <td align="center">22.372</td>
    </tr>
    <tr>
      <td align="center">Original</td>
      <td align="center">Original</td>
      <td align="center">42.497</td>
      <td align="center">26.182</td>
      <td align="center">22.378</td>
    </tr>
  </tbody>
</table>
"""

LATEX_TABLE = """
        & original & 30.23 $\pm$ 0.09 & 54.79 $\pm$ 0.06 & 28.72 $\pm$ 0.15 & 7.23 $\pm$ 0.04 & 30.49 $\pm$ 0.14 & 51.23 $\pm$ 0.07 & 42.41 $\pm$ 0.07 & 25.96 $\pm$ 0.15 & 22.32 $\pm$ 0.04 \\
        & DNAT & 30.28 $\pm$ 0.03 & 54.86 $\pm$ 0.09 & 28.82 $\pm$ 0.08 & 7.15 $\pm$ 0.13 & 30.57 $\pm$ 0.06 & 51.31 $\pm$ 0.08 & 42.48 $\pm$ 0.02 & 25.90 $\pm$ 0.14 & 22.44 $\pm$ 0.03 \\
        & blur & 30.17 $\pm$ 0.06 & 54.66 $\pm$ 0.10 & 28.83 $\pm$ 0.08 & 7.24 $\pm$ 0.02 & 30.50 $\pm$ 0.10 & 51.08 $\pm$ 0.10 & 42.41 $\pm$ 0.04 & 25.87 $\pm$ 0.13 & 22.22 $\pm$ 0.04 \\
    """


def main():
    metrics = defaultdict(lambda: defaultdict(list))
    for mode in MODES:
        for iter in range(N_ITERATIONS):
            folder = os.path.join(ROOT, f"{NAME}_{mode}_{iter}")
            for cls in CLASSES:
                file = os.path.join(folder, f"{cls}_coco_results.json")
                with open(file, "r") as f:
                    data = json.load(f)
                metrics[mode][cls].append(data)

    # make all metrics numpy arrays
    for mode in MODES:
        for cls in CLASSES:
            metrics[mode][cls] = np.array(metrics[mode][cls])
    # make all metrics into mean and std
    for mode in MODES:
        for cls in CLASSES:
            metrics[mode][cls] = np.mean(metrics[mode][cls], axis=0), np.std(
                metrics[mode][cls], axis=0
            )
    # recreate the table
    for mode in MODES:
        ap, ap_std = metrics[mode]["All"][0][0], metrics[mode]["All"][1][0]
        ap50, ap50_std = metrics[mode]["All"][0][1], metrics[mode]["All"][1][1]
        ap75, ap75_std = metrics[mode]["All"][0][2], metrics[mode]["All"][1][2]
        aps, aps_std = metrics[mode]["All"][0][3], metrics[mode]["All"][1][3]
        apm, apm_std = metrics[mode]["All"][0][4], metrics[mode]["All"][1][4]
        apl, apl_std = metrics[mode]["All"][0][5], metrics[mode]["All"][1][5]
        ap_veh, ap_veh_std = (
            metrics[mode]["Vehicle"][0][0],
            metrics[mode]["Vehicle"][1][0],
        )
        ap_vuln, ap_vuln_std = (
            metrics[mode]["VulnerableVehicle"][0][0],
            metrics[mode]["VulnerableVehicle"][1][0],
        )
        ap_ped, ap_ped_std = (
            metrics[mode]["Pedestrian"][0][0],
            metrics[mode]["Pedestrian"][1][0],
        )

        #         print(
        #             f"""<tr>
        # <td align="center">{mode}</td>
        # <td align="center">{ap*100:.3f} &plusmn; {ap_std*100:.3f}</td>
        # <td align="center">{ap50*100:.3f} &plusmn; {ap50_std*100:.3f}</td>
        # <td align="center">{ap75*100:.3f} &plusmn; {ap75_std*100:.3f}</td>
        # <td align="center">{aps*100:.3f} &plusmn; {aps_std*100:.3f}</td>
        # <td align="center">{apm*100:.3f} &plusmn; {apm_std*100:.3f}</td>
        # <td align="center">{apl*100:.3f} &plusmn; {apl_std*100:.3f}</td>
        # <td align="center">{ap_veh*100:.3f} &plusmn; {ap_veh_std*100:.3f}</td>
        # <td align="center">{ap_vuln*100:.3f} &plusmn; {ap_vuln_std*100:.3f}</td>
        # <td align="center">{ap_ped*100:.3f} &plusmn; {ap_ped_std*100:.3f}</td>
        # </tr>"""
        #         )

        print(
            f"""        & {mode} & {ap*100:.2f} $\pm$ {ap_std*100:.2f} & {ap50*100:.2f} $\pm$ {ap50_std*100:.2f} & {ap75*100:.2f} $\pm$ {ap75_std*100:.2f} & {aps*100:.2f} $\pm$ {aps_std*100:.2f} & {apm*100:.2f} $\pm$ {apm_std*100:.2f} & {apl*100:.2f} $\pm$ {apl_std*100:.2f} & {ap_veh*100:.2f} $\pm$ {ap_veh_std*100:.2f} & {ap_vuln*100:.2f} $\pm$ {ap_vuln_std*100:.2f} & {ap_ped*100:.2f} $\pm$ {ap_ped_std*100:.2f} \\\\"""  # noqa
        )


if __name__ == "__main__":
    main()
