#### �ۑ�1
- `./src/prob1.py` �ɉۑ�1�ɑΉ�����֐�����`����Ă���. `agg1`, `agg2`, `READOUT` �����ꂼ��W��-1, �W��-2, READOUT �ɑΉ����Ă���.
- �ۑ�1�ɑ΂���e�X�g��, `./src/`����`python prob1_test.py`�����s���邱�Ƃœ�����. �e�X�g�ł�`generate_graph` �ɂ�萶�����ꂽ$3\times 3$�̃����_���ȃO���t�ɑ΂���, �W���READOUT�̌��ʂ������o���Ă���. �����ʎ��� $D$ �ƏW��� $T$ �͍ŏ��ɒ�`���Ă���. $W$ �̑S�Ă̗v�f��1, �������͑S�Ă̗v�f��-1��2�̏ꍇ�̃e�X�g��p�ӂ��Ă���. �����ReLU ���@�\���Ă��邩���m���߂邽�߂ł���. ���ʂ̐������͏o�͂��ꂽ�O���t�����v�Z�Ŋm���߂�.

#### �ۑ�2
- `./src/prob2.py` �ɃO���t $G$ �Ƃ��̃��x�� y_true �ɑ΂���, ����L(loss)�ƃ��x���̗\����Ԃ��֐�, `calc_GNN` ���`����.
- `./src/` �ɂ�����`python prob2-2.py` �����s�����, �C�ӂ�$N\times N$�̃O���t�ƃ��x���ɑ΂�����z�~���@�̌��ʂ��m�F�ł���. �X�N���v�g�ł�$N=13$ �Ƃ��Ă��邪����$N$���w�肵�Ă����Ȃ�.

#### �ۑ�3
- `./src/`�ɂ� `python prob3_sgd.py`�����s���邱�Ƃ� SGD�ł�GNN�̌P�����s����. �P���f�[�^�̏ꏊ��, `TrainData_path = './machine_learning/datasets/train/'`�Ƃ��Ă��� (�ȉ����l). ���s��ɂ͊eepoch�ɂ�����w�K�f�[�^�ƌ���p�f�[�^�̕��ϑ����ƕ��ϐ��x���v���b�g����� (�ȉ����l).
- `./src/`�ɂ� `python prob3_msgd.py`�����s���邱�Ƃ� Momentum SGD�ł�GNN�̌P�����s����. 

#### �ۑ�4
- `./src/`�ɂ� `python prob4_adam.py`�����s���邱�Ƃ� Adam�ɂ��GNN�̌P�����s����. 
- `./src/`�ɂ� `python prob4_prediction.py`�����s���邱�Ƃ�Adam�ɂ���ē���ꂽ�p�����[�^�ɂ��e�X�g�f�[�^�̗\�����s����. �\���ɂ�`./src/data/adam100_W.npy`�ȂǂɊi�[���ꂽ$W,A,b$�̃f�[�^(100epochs�ڂł̃p�����[�^)���K�v�ł���. �e�X�g�f�[�^�̏ꏊ��`TestData_path = './machine_learning/datasets/test/'`�Ƃ��Ă���.