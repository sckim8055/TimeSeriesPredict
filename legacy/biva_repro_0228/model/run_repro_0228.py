"""2026-02-28 사용자 실행((BASE)BIVA_IlWonSoh_20260227.ipynb) 재현. 데이터 QualterlyCleaned_20260227_2.xlsx, 타겟 B1, 6채널, lr 0.1.
원본 코드·원본 기간(1998-01) 그대로. 경로·장치만 패치. 기대값(사용자 결과): RMSE 1.3033 / 1.3989 (두 번 실행)."""
import sys, os, random, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, torch
torch.set_num_threads(1)
from exp_main import Exp_Main
fix_seed = 2025; random.seed(fix_seed); torch.manual_seed(fix_seed); np.random.seed(fix_seed)
class args: pass
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
args.is_training=True; args.model='BIVA'; args.model_id='REPRO0228-seq6-pred1-bt3-lr0.1'
args.data='BIVA'; args.root_path=os.path.join(HERE,'..','Base_data'); args.data_path='QualterlyCleaned_20260227_2.xlsx'
args.exp_dir=os.path.join(HERE,'..','exp'); os.makedirs(args.exp_dir, exist_ok=True)
args.des='Exp'; args.loss='mse'; args.lradj='3'; args.use_amp=False
args.features='MS'; args.target='B1'; args.freq='M'; args.checkpoints=os.path.join(args.exp_dir,'checkpoints')+'/'
args.seq_len=6; args.pred_len=1; args.label_len=1
args.channels=6; args.rnn_hid_size=128; args.vae_hid_size=128; args.vae_latent_size=3; args.infer_hid_size=128
args.moving_avg=4; args.cols=None; args.conv1d=True; args.conv_kernal=1; args.RIN=False; args.combination=True
args.imputed_weight=0.05; args.last_chkpt=None
args.learning_rate=0.1; args.batch_size=3; args.train_epochs=epochs; args.patience=500
args.embed='timeF'; args.inverse=False; args.do_predict=False
args.num_workers=0; args.itr=1; args.use_gpu=False; args.gpu=0; args.use_multi_gpu=False; args.devices='0'
t0=time.time(); exp=Exp_Main(args); exp.train('repro'); preds,trues,inputx,mae,mse,rmse,mape,mspe,rse,corr=exp.test('repro')
out=dict(variant='repro-20260228-IlWonSoh', expected_user_rmse=[1.3032854,1.3988984], mae=float(mae), mse=float(mse), rmse=float(rmse), mape=float(mape), n_test=int(preds.shape[0]), epochs=epochs, seconds=round(time.time()-t0,1))
print(json.dumps(out)); json.dump(out, open(os.path.join(args.exp_dir,'summary_repro0228.json'),'w'), indent=1)
