# AIかわいい判定ラボ

Oxford-IIIT Pet Dataset の画像を使い、参加者が「かわいい！」「今回はちがう」とラベルを付けながら、自分好みの“かわいい判定AI”を育てる Human-in-the-Loop 型の3時間ワークショップ教材です。

当日は深層学習モデルの再学習をしません。講師が事前に画像 embedding を作成し、参加者は `metadata.csv` と `embeddings.npy` を読み込んで、軽量な `LogisticRegression` だけを更新します。

## 構成

```text
cute-ai-workshop/
  README.md
  requirements.txt
  pyproject.toml
  uv.lock
  .python-version
  scripts/
    prepare_dataset.py
    create_embeddings.py
    make_workshop_package.py
  notebooks/
    01_prepare_dataset.ipynb
    02_participant_workshop.ipynb
  data/
    workshop/
      images/
      metadata.csv
      embeddings.npy
  docs/
    instructor_notes.md
    mini_lecture_slides.md
```

## 事前準備

ローカルでは **uv + Python 3.12 + JupyterLab** を使います。初回はネット接続が必要です。
インストールから当日の起動までの詳細は [ローカル環境構築手順](docs/local_setup.md) を参照してください。
依存関係は `pyproject.toml`、固定バージョンは `uv.lock` で管理します。

```bash
cd cute-ai-workshop
uv sync --locked --extra prepare
```

## 1. Oxford-IIIT Pet Dataset を取得して小さな教材データを作る

```bash
uv run --locked --extra prepare python scripts/prepare_dataset.py --root-dir . --total-images 240
```

作成されるもの:

- `data/workshop/images/`
- `data/workshop/metadata.csv`

初期設定では、犬猫の両方を含む12品種から合計約240枚を選びます。`metadata.csv` には `species` と `breed` が入ります。

## 2. 画像 embedding を作る

```bash
uv run --locked --extra prepare python scripts/create_embeddings.py --root-dir .
```

作成されるもの:

- `data/workshop/embeddings.npy`

CUDA → MacのMPS（Metal）→ CPU の順で、利用できるデバイスを自動選択します。起動時の `Using device: mps` などの表示で確認できます。CPU だけで実行したい場合は次のようにします。

```bash
uv run --locked --extra prepare python scripts/create_embeddings.py --root-dir . --cpu
```

## 3. Colab 配布用 zip を作る

```bash
uv run --locked python scripts/make_workshop_package.py --root-dir .
```

`cute-ai-workshop-package.zip` が作られます。Colab で使う場合は、この zip を Google Drive に置くか、Colab にアップロードして展開してください。
ローカルPCにも同じzipを配布できます。仮想環境や取得元の全画像は含めません。

## ローカルでの起動

プロジェクトフォルダで実行します。参加者には PyTorch / GPU は不要です。

```bash
uv sync --locked
uv run --locked python -m ipykernel install --sys-prefix --name cute-ai-uv --display-name "Cute AI (uv)"
uv run --locked jupyter lab
```

参加者ノートブックを開き、カーネルは **Cute AI (uv)** を選びます。
次回からは最後の起動コマンドだけで使えます。

## 参加者用ノートブック

参加者は `notebooks/02_participant_workshop.ipynb` を開きます。

ノートブックでできること:

- 画像を1枚ずつ表示
- 「かわいい！」「今回はちがう」「スキップ」でラベル付け
- 「AIを更新する」で軽量分類器を再学習
- 学習後に「AIかわいい度」を表示
- 「AIの推しTOP10」を画像付きで表示
- 慎重なAI、ふつうのAI、思い込み強めAIを比較
- 6-Bでロジスティック回帰と小さな3層MLPを比較（メインAIは変更しません）
- species / breed ごとのAIかわいい度を集計
- t-SNEの固定地図と、元の分類器の判定スコア図を表示（6-Aのセル）
- ラベル済み枚数、かわいい枚数、今回はちがう枚数を表示
- 出題モードを選択
  - ランダム
  - AIが迷っている画像
  - 自動
- ラベルをリセット
- ラベル結果を CSV 保存

## ワークショップで扱う機械学習のポイント

この教材では、次の内容を体験として扱います。

- 画像をそのまま学習するのではなく、事前学習済みモデルで特徴量に変換する
- 少量ラベルから、軽量な分類器が好みの境界線を学ぶ
- 「かわいい！」と「今回はちがう」の両方がないと学習できない
- ラベルが少ないと、AIの判断は不安定になりやすい
- 教えた例に合わせすぎると、まだ見ていない画像にも強く決めつけることがある

参加者ノートブックには「思い込み強めAI」の比較実験があります。これは過学習を直感的に扱うための実験です。厳密な評価指標よりも、少ない例だけを信じすぎるとAIの推し方が変わることを観察します。

## 確認方法

事前準備後、次を確認してください。

```bash
ls data/workshop/images | head
ls data/workshop/metadata.csv
ls data/workshop/embeddings.npy
uv run --locked python - <<'PY'
import numpy as np
import pandas as pd
meta = pd.read_csv("data/workshop/metadata.csv")
emb = np.load("data/workshop/embeddings.npy")
print(meta.shape)
print(emb.shape)
assert len(meta) == emb.shape[0]
assert emb.shape[1] == 1280
PY
```

Colab または Jupyter で `notebooks/02_participant_workshop.ipynb` を開き、10枚程度ラベルを付けてから「AIを更新する」を押してください。「かわいい！」と「今回はちがう」の両方が1枚以上あれば学習できます。

## 可視化の確認

6-Aのセルを実行すると、画像の地図と判定スコア図を表示します。画像No.を選ぶと実画像と図中の位置を確認できます。追加ラベルで「AIを更新する」を押し、地図の配置は固定のまま色が変わること、スコア図の横位置が変わることを確認してください。リセット後は学習を促すメッセージが表示されます。

判定は元の1280次元の特徴量を使います。t-SNEには境界線を引きません。スコア図の横軸は `decision_function`（標準化した特徴量に対する `wᵀz+b`）、0が境界です。縦軸は固定したランダムな散らし位置です。これは超平面の断面図や距離を保存した地図ではありません。

参考: [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)、[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## 注意

### GitHubへの公開

コード・説明書・出力を消したノートブック・uv設定を管理します。生成済みの `data/workshop/`、取得元の `data/raw/`、配布zip、参加者の自動保存JSON・バックアップ・提出用CSV、仮想環境は `.gitignore` で除外しています。これらは手元には残ります。クローン後は事前準備の手順で教材データを生成してください。

ノートブックを実行してから公開する際は、全セルの出力と実行番号を消して保存してください。`.gitignore` はノートブック内部の出力を除外するものではありません。

画像データはコードとは別の著作物です。[Oxford-IIIT Pet Dataset公式ページ](https://www.robots.ox.ac.uk/~vgg/data/pets/)にはCC BY-SA 4.0と記載されています。画像を含む教材zipを再配布する際は、出典・ライセンス表示などの条件を確認してください。

### MLP比較（6-B）

参加者ノートブックの6-Bセルを実行し「比較する」を押します。同じ特徴量と同じ学習ラベルで、実験専用のロジスティック回帰とMLPを新規学習します。MLPは入力→隠れ16ユニット（ReLU）→二値出力、L2正則化alpha=1、L-BFGS、乱数seed=42です。CPUで実行し、画像特徴抽出モデルは再学習しません。

3-Aで固定確認用への回答を完了し、学習用に両クラス各1枚以上教えれば実行可能です。全モデルが同じ固定確認用セットを使います。標準化は学習分のみ、MLPはsample_weightでクラスの偏りを補正します（scikit-learn 1.7以上）。学習・確認用一致率、収束状態、学習時間、未ラベルTOP5、予測差が大きい画像番号を表示します。

確認用は画像パスの固定ハッシュ順で選ぶ最大20枚（少数データでは約20%）です。学習枚数が増えても同じ画像を評価します。少数の確認用画像への一致率は参考値で、繰り返し結果を見て調整すれば確認用への過適合も起こります。学習一致率や予測の強さだけで過学習とは判定しません。

比較はメインのmodel・scores・labelsを変更しません。比較の評価結果は履歴に追加されます。環境更新は `uv sync --locked` 後にカーネルを再起動してください（先にラベルをCSV保存）。開いているノートブックにはディスク上の更新を読み直す必要があります。

### 固定確認セットで10枚・20枚・40枚を比較

1. 講師が最初に「すべてのセルを実行」で画面を準備します。
2. 3-AでAI予測を見る前に確認用全枚へ回答します。この回答は学習に混ぜません。
3. 4で学習用10枚に回答し「この枚数で比較・記録」を押します。通常AI・設定違い・MLPの比較をまとめて記録します。
4. 推しTOP10や「記録を見る」で結果を確認し、「次の段階へ」で累計20枚、40枚へ進みます。出題はランダム固定です。両方の回答がそろわない場合は追加で回答でき、実際の枚数で記録します。
5. CSV保存で回答と評価履歴を提出用に書き出します。

回答・履歴は教材直下の `participant_progress.json` に自動保存します。セルを再実行しても保持し、再起動後も上から実行すれば復元します。学習済みモデルはファイルに保存しないため、再起動後は「AIを更新する」で再学習します。

「ラベルをリセット」は確認用回答を保持します。「新しい参加者」は確認用も初期化します。どちらも確認後に日時付きJSONバックアップを作ってから実行します。保存が壊れている・教材が異なる場合は上書きせずエラーを表示します。自動保存の失敗は画面に表示します。CSVからの復元は未対応です。

配布スクリプトは教材直下の自動保存・バックアップJSONを含めません。手動で教材フォルダをコピーする際は `participant_progress*.json` を除いてください。複数のノートブック・カーネルで同じ教材を同時に操作しないでください。

- 参加者用ノートブックでは深層学習モデルを再学習しません。
- AIは正解を当てるものではなく、人の好みを少しずつ学ぶものとして扱います。
- 否定的な表現を避けるため、教材内では「かわいくない」ではなく「今回はちがう」を使います。

### 40枚より先の追加学習

40枚の比較が終わったら、4の「自由に追加して学習する」を押せます。以後は残りの学習用写真に自由に回答し、「AIを更新する」で再学習します。「この枚数で比較・記録」も任意の枚数で使えます。10・20・40枚の比較記録と固定確認用の回答は保持します。自由追加の状態も自動保存・復元します。学習用写真にすべて回答すると回答ボタンは無効になります。
