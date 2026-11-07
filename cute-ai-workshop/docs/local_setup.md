# ローカル環境構築: uv + JupyterLab

Googleアカウントは不要です。ブラウザは自分のPC上のJupyterLabにつながります。
参加者用はCPUで動作します。画像特徴量を作る講師用だけ、追加でPyTorchを入れます。

## 1. uvを入れる（PCごとに1回）

macOS / Linuxのターミナル:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

WindowsのPowerShell（WinGetが利用できる場合）:

```powershell
winget install --id=astral-sh.uv -e
```

ターミナルを開き直して `uv --version` が表示されることを確認します。
別の導入方法は [uv公式](https://docs.astral.sh/uv/getting-started/installation/) を参照してください。

## 2. 教材フォルダに移動

macOSでDocumentsに教材フォルダを置いた場合の例です。実際の保存場所に合わせて変更してください。

```bash
cd "$HOME/Documents/cute-ai-workshop"
```

Windowsなど別のPCでは、配布zipを展開し、`pyproject.toml` があるフォルダをターミナルで開きます。
以下のuvコマンドはMac / Windows共通です。Python 3.12は必要に応じてuvが取得します。
以前のvenvやcondaを有効化している場合は、新しいターミナルで始めてください。
古い `.venv-local` は削除不要です。uvはこのフォルダの `.venv` を使います。

## 3. 環境を準備

参加者（生成済みデータを使う場合）:

```bash
uv sync --locked
```

講師（データ取得・特徴量生成もする場合）:

```bash
uv sync --locked --extra prepare
```

この2つはどちらか一方で構いません。`--extra prepare` がPyTorch等を追加します。
`requirements.txt` はColab等の従来手順向けです。ローカルのuv環境ではpipで追加インストールしません。
カーネルを環境内に登録します（初回のみ）。

```bash
uv run --locked python -m ipykernel install --sys-prefix --name cute-ai-uv --display-name "Cute AI (uv)"
uv run --locked python -c "import numpy, pandas, sklearn, matplotlib, ipywidgets; print('Environment OK')"
```

## 4. 教材データを用意（講師のみ・事前に1回）

```bash
uv run --locked --extra prepare python scripts/prepare_dataset.py --root-dir . --total-images 240
uv run --locked --extra prepare python scripts/create_embeddings.py --root-dir .
```

ダウンロードと特徴量生成には時間がかかります。当日までに完了してください。
次の3つが必要です。

- `data/workshop/images/`
- `data/workshop/metadata.csv`
- `data/workshop/embeddings.npy`

CUDA → MacのMPS（Metal）→ CPU の順で自動選択します。Macで `Using device: mps` と表示されればGPUを使用しています。
CPUで実行したい場合やGPUでエラーが出る場合は、コマンド末尾に `--cpu` を付けて実行してください。
NVIDIA GPUを使う場合はOS・ドライバに合うPyTorchの設定が別途必要になることがあります。
GPU用のインストールを変える際もuvの設定・lockを更新し、実機確認してください。

配布zipの作成:

```bash
uv run --locked python scripts/make_workshop_package.py --root-dir .
```

コード・設定・lock・教材データを配ります。`.venv` 自体は配らず、各PCで手順3を実行します。

## 5. 起動と操作

初回の環境構築・カーネル登録・教材データの準備が終わったら、参加者は教材フォルダ内の起動ファイルをダブルクリックするだけで起動できます。

- Windows: `start_workshop.bat`
- Mac: `start_workshop.command`

どちらも、そのファイルのあるフォルダで `uv run --locked jupyter lab` を実行します。ファイルだけを別の場所に移さず、教材フォルダごと配布してください。Windowsではショートカット、Macではエイリアスをデスクトップに置いても構いません。作業中は起動したターミナルを閉じないでください。

Macで実行権限のエラーになる場合は、講師が教材フォルダで一度 `chmod +x start_workshop.command` を実行してください。起動ファイルも配布zipに含まれます。手動起動する場合は以下です。

```bash
uv run --locked jupyter lab
```

ブラウザが開かなければ、ターミナルに表示されたローカルURLを開いてください。
`notebooks/02_participant_workshop.ipynb` を開き、カーネル選択で **Cute AI (uv)** を選びます。
講師が最初に「すべてのセルを実行」で画面を準備します。参加者は3-Aと4のボタンだけで進められます。ローカルではパッケージ準備セルはインストールを行いません。
データの場所は、プロジェクト直下または `notebooks/` から自動検出します。
見つからない場合は `WORKSHOP_DIR` に実際のデータフォルダの絶対パスを指定してください。

確認項目:

1. 画像とラベル付けボタンが表示される。
2. 両方のラベルを含めて約10枚教え、「AIを更新する」が動く。
3. TOP10に画像と予測が表示される。
4. 6-Aを実行し、地図・判定スコア図・画像番号によるプレビューが表示される。
5. 追加ラベルで再学習すると図が更新される。
6. CSV保存後、JupyterLabのファイル一覧に `my_cute_ai_labels.csv` が見える（通常はnotebooks内）。

回答と履歴は教材直下の `participant_progress.json` に自動保存されます。セルの再実行で回答は消えず、カーネル再起動後も上から実行すると復元します（AIは再学習が必要です）。提出用にはCSV保存も行ってください。別の参加者に交代する際は「新しい参加者」から確認してリセットします。日時付きJSONバックアップを残します。

10枚・累計20枚・累計40枚で回答をいったん止め、「この枚数で比較・記録」→「次の段階へ」で進みます。出題はランダム固定です。片方の回答しかない場合は追加で回答し、実際の枚数で記録します。
終了時はJupyterLabを起動したターミナルでCtrl+Cを押し、終了確認に従います。

## 6. 当日・次回

教材フォルダで `uv run --locked jupyter lab` を実行します。
完全オフライン運用は、事前セットアップ後に次のコマンドで実機リハーサルしてください。

```bash
uv run --offline --locked jupyter lab
```

未取得のPythonや依存パッケージはオフラインでは取得できません。
本番前に全PCで上の確認項目を試してください。

## トラブルと更新

- `uv` が見つからない: インストール後にターミナルを開き直す。
- importエラー: ノートブックで `import sys; print(sys.executable)` を実行し、この教材の `.venv` 内か確認する。
- ボタンが表示されない: Cute AI (uv) を選び直し、カーネルを再起動して上から実行する。
- `--locked` で不整合エラー: 配布した `pyproject.toml` と `uv.lock` が同じ版か確認する。参加者側でlockを作り直さない。
- 講師の依存追加: `uv add パッケージ名`。準備専用なら `uv add --optional prepare パッケージ名`。
- 一括更新: 講師が `uv lock --upgrade` → `uv sync --locked --extra prepare` → 動作確認してから配布する。

lockは依存バージョンを固定しますが、OSやGPUドライバの違いまではなくしません。
Pythonのマイナーバージョンは `.python-version` と `requires-python` で3.12にそろえています。
参考: [uvプロジェクト管理](https://docs.astral.sh/uv/guides/projects/)、[Jupyter連携](https://docs.astral.sh/uv/guides/integration/jupyter/)。

### 確認用で判断が違った画像を見る

4の「この枚数で比較・記録」の後、「判断の違う画像を見る」を押します。「評価記録」でモデルと学習枚数を選ぶと、その評価時点で自分の回答とAI予測が食い違った画像・回答・予測・AIかわいい度を表示します。全枚一致の場合はその旨を表示します。追加の学習や回答の書き換えは行いません。

画像別の結果も自動保存され、再起動後も参照できます。旧版の評価記録には画像別予測がないため、一覧には出ません。既存の一致率履歴を保持したまま、新たに比較・記録してください。結果を見て教え方を改善する場合、その後の確認用一致率は独立した最終テストではなく改善用の評価です。

### 40枚より先の追加学習

40枚の比較が終わったら、4の「自由に追加して学習する」を押せます。以後は残りの学習用写真に自由に回答し、「AIを更新する」で再学習します。「この枚数で比較・記録」も任意の枚数で使えます。10・20・40枚の比較記録と固定確認用の回答は保持します。自由追加の状態も自動保存・復元します。学習用写真にすべて回答すると回答ボタンは無効になります。
