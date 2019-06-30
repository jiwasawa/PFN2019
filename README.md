#### 課題1
- `./src/prob1.py` に課題1に対応する関数が定義されている. `agg1`, `agg2`, `READOUT` がそれぞれ集約-1, 集約-2, READOUT に対応している.
- 課題1に対するテストは, `./src/`内で`python prob1_test.py`を実行することで得られる. テストでは`generate_graph` により生成された$3\times 3$のランダムなグラフに対して, 集約とREADOUTの結果を書き出している. 特徴量次元 $D$ と集約回数 $T$ は最初に定義している. $W$ の全ての要素が1, もしくは全ての要素が-1の2つの場合のテストを用意している. これはReLU が機能しているかを確かめるためである. 結果の整合性は出力されたグラフから手計算で確かめた.

#### 課題2
- `./src/prob2.py` にグラフ $G$ とそのラベル y_true に対して, 損失L(loss)とラベルの予測を返す関数, `calc_GNN` を定義した.
- `./src/` において`python prob2-2.py` を実行すると, 任意の$N\times N$のグラフとラベルに対する勾配降下法の結果を確認できる. スクリプトでは$N=13$ としているが他の$N$を指定しても問題ない.

#### 課題3
- `./src/`にて `python prob3_sgd.py`を実行することで SGDでのGNNの訓練が行われる. 訓練データの場所は, `TrainData_path = './machine_learning/datasets/train/'`としている (以下同様). 実行後には各epochにおける学習データと検定用データの平均損失と平均精度がプロットされる (以下同様).
- `./src/`にて `python prob3_msgd.py`を実行することで Momentum SGDでのGNNの訓練が行われる. 

#### 課題4
- `./src/`にて `python prob4_adam.py`を実行することで AdamによるGNNの訓練が行われる. 
- `./src/`にて `python prob4_prediction.py`を実行することでAdamによって得られたパラメータによるテストデータの予測が行われる. 予測には`./src/data/adam100_W.npy`などに格納された$W,A,b$のデータ(100epochs目でのパラメータ)が必要である. テストデータの場所は`TestData_path = './machine_learning/datasets/test/'`としている.