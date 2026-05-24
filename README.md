# CPST Tea-Making Stage Recognition

This workspace implements the updated paper's CPST fusion flow on a fine-grainedtea-making task. The target is not just "making tea", but which stage theperson is currently performing.

## Stages

1. Walk to kitchen / tea area
2. Take cup / teapot
3. Add tea leaves / tea bag
4. Add water
5. Heat / wait
6. Pour into cup
7. Drink / carry away

## Paper-To-Code Mapping

The script follows the paper's four-space formula:

    X_t = {X_t^P, X_t^C, X_t^S, X_t^T}
    h_t^P = f_P(X_{t-k:t}^P)
    h_t^C = f_C(X_t^C, H_t)
    h_t^S = f_S(G_t^S)
    h_t^T = f_T(X_t^T)
    u_t^i = g_i(h_t^i)
    alpha_t^i = softmax(-u_t^i)
    A_ij = softmax(((Qh_i)(Kh_j)^T) / sqrt(d))
    y_hat = softmax(W_o z_tilde + b_o)

Implementation choices:

* `P`: GRU temporal encoder over simulated IMU/location/vision/contact/water-flow windows.
* `C`: MLP encoder over smart-kettle IoT, energy, RFID, app, and history logs.
* `S`: graph encoder over user/guest/family/shared-context social graph `G=(V,E)`.
* `T`: MLP encoder over intention, habit, preference, cognitive load, and micro-stage plan.
* `M9`: adds uncertainty calibration with `L = L_cls + lambda L_unc + beta L_reg`.

## Run

    & 'D:\miniconda\envs\myenv\python.exe' .\cpst_tea_stage_experiment.py

The default experiment uses 520 simulated tea-making episodes and splits byepisode, so samples from the same episode do not leak across train/validation/test.

## Ablations

* `M1_Physical`: Physical only
* `M2_Cyber`: Cyber only
* `M3_Social`: Social only
* `M4_Thinking`: Thinking only
* `M5_P+C`: Physical + Cyber
* `M6_P+C+S`: Physical + Cyber + Social
* `M7_CPST_Attention`: four-space cross-attention fusion
* `M8_CPST_Concat`: four-space simple concatenation baseline
* `M9_CPST_Uncertainty`: four-space attention plus uncertainty-aware weighting

## Outputs

Results are written to `tea_results/`:

* `tea_ablation_metrics.csv`: accuracy, macro-F1, early-stage accuracy.
* `tea_per_stage_accuracy.csv`: per-stage accuracy for all models.
* `tea_fusion_weights.csv`: learned confidence, uncertainty, and attention summaries.
* `tea_synthetic_samples.csv`: sample simulated CPST observations.
* `tea_ablation_accuracy.png`: overall vs early-stage accuracy chart.
* `tea_confusion_matrices.png`: normalized confusion matrix comparison.
* `tea_experiment_summary.json`: feature lists and formula mapping.

The earlier generic ADL script is kept as `cpst_adl_ablation.py`; the updatedpaper-specific tea experiment is `cpst_tea_stage_experiment.py`.

## Contrast Activities

`cpst_contrast_activity_experiment.py` adds easily confused activities aroundtea making:

* making tea
* making coffee
* drinking water
* cooking
* washing a cup
* serving tea to a guest

This experiment checks whether the model can avoid treating every cup/waterinteraction as tea making. It reports normal accuracy, early-activity accuracy,and `make_tea_false_positive_rate`, the rate at which non-tea activities aremistaken for plain tea making.

Run:

    & 'D:\miniconda\envs\myenv\python.exe' .\cpst_contrast_activity_experiment.py

Outputs are written to `contrast_results/`:

* `contrast_ablation_metrics.csv`
* `contrast_per_activity_accuracy.csv`
* `contrast_fusion_weights.csv`
* `contrast_synthetic_samples.csv`
* `contrast_ablation_accuracy.png`
* `contrast_confusion_matrices.png`

## Robustness Under Missing Spaces

`cpst_robustness_experiment.py` tests the contrast-activity task undertest-time data failures:

* Physical missing: camera occlusion or wearable not worn.
* Cyber missing: smart kettle / IoT logs offline.
* Social missing: no dialogue, visitor, or relation context.
* Thinking missing: no history, preference, or schedule information.
* Multi-space noise: sensor false alarms plus delayed/noisy device logs.

Run:

    & 'D:\miniconda\envs\myenv\python.exe' .\cpst_robustness_experiment.py

Outputs are written to `robustness_results/`:

* `robustness_metrics.csv`
* `robustness_key_comparison.csv`
* `robustness_macro_f1_pivot.csv`
* `robustness_macro_f1_drop_pivot.csv`
* `robustness_macro_f1.png`
* `robustness_full_cpst_drop.png`

## Attention And Uncertainty

`cpst_attention_uncertainty_experiment.py` adds two checks from the CPSTmethod section:

* Cross-space attention interpretability: exports stage-by-stage attentionreceived from the matrix `A` and stage-level fusion confidence.
* Uncertainty-aware fusion: evaluates`alpha_t^i = exp(-u_t^i) / sum_j exp(-u_t^j)` under missing/noisy spaces.

Run:

    & 'D:\miniconda\envs\myenv\python.exe' .\cpst_attention_uncertainty_experiment.py

Outputs are written to `attention_uncertainty_results/`:

* `attention_stage_fusion_confidence.csv`
* `attention_stage_received.csv`
* `attention_stage_cross_space_matrix.csv`
* `attention_stage_top_spaces.csv`
* `attention_stage_top_attention_received.csv`
* `attention_stage_fusion_confidence_heatmap.png`
* `attention_stage_received_heatmap.png`
* `uncertainty_comparison_metrics.csv`
* `uncertainty_confidence_weights_heatmap.png`
