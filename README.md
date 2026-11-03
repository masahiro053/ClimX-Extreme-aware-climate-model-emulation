# ClimX-Extreme-aware-climate-model-emulation

## 🎯 コンペ概要 (Overview)
K-12（幼稚園〜高校卒業レベル）の生徒が解いた**数学の手書きノート画像**を分析し、生徒が計算の過程でどのような「根本的な誤解（Misconception）」をしているかを特定するコンペティション。

- **Input**: 問題文 (`question_text`)、トピック (`topic_name`)、生徒の手書きノート画像 (`image_file`)
- **Target**: 公式が定義した75種類の「誤解コード」から該当するものを予測。完全に正解している（誤解がない）場合は `NONE` とする。
- **Evaluation**: **MAP@3** (Mean Average Precision @ 3)。1つの解答（`trace_id`）につき、確信度の高い順に最大3つのエラーコード（スペース区切り）を提出する。
- **Constraints**: Kaggle Notebookでの推論必須（Internet: OFF）。実行時間制限9時間。

---

## 2026-11-03
**Score**: LB 0.47916

### Done
- **環境構築とVLM推論パイプラインの開通**
  - Kaggle特有のディレクトリ構造トラップに対し、`os.walk` を用いた画像パスの完全マッピングを実装して回避。
  - Kaggleオフライン環境での量子化モデル（AWQ/GPTQ）ライブラリ依存エラーに遭遇。標準の `qwen2-vl-2b-instruct` (非量子化モデル) に切り替えて解決し、OOM対策として `image.thumbnail((512, 512))` を導入。
  - submissionのフォーマットエラー (`QuestionId` -> `trace_id`, `Target` -> `root_code`) を修正し、初提出を完走。
- **経験ベイズ法（Empirical Bayes）の導入**
  - 公開Notebookをベースに、VLMの出力スコアに事前分布（Prior）を掛け合わせる後処理を実装。
  - `train.csv` の `topic_name` と `root_code` からトピックごとの出現確率を計算。ラプラス平滑化（`alpha = 0.5`）を適用し、未知のトピックにも対応。

### Insights
- **VLMの幻覚（Hallucination）抑制には事前分布が有効**: 代数の問題で幾何のエラーコードを出力してしまうようなVLMの弱点を、`topic_name` ベースの経験ベイズ補正で強力に抑制できることを確認。
- **Kaggle環境の落とし穴**: VLMのロードには `AutoModelForCausalLM` ではなく `Qwen2VLForConditionalGeneration` が必要。また、オフライン推論での外部ライブラリ (`optimum` 等) 要求は致命的になるため、モデル選定時は要注意。

### TODO
- [ ] `alpha` (平滑化パラメータ) のCVチューニング。
- [ ] DataLoaderを用いたバッチ推論（Batch Inference）の実装による推論時間の短縮。
- [ ] Test-Time Augmentation (TTA): 画像解像度（例: 512px, 768px）を変更した複数回推論のアンサンブル。